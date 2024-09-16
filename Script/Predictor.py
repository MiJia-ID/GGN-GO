import os
import csv
import glob
import json

import time
from utils import extract_sequences_from_fasta, save_sequences_to_fasta
from model import *
import torch
from utils import load_GO_annot_prog, log, extract_single_protein
import argparse
from config import get_config
from myData import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import warnings


def str2bool(v):
    """Convert a string to a boolean value."""
    if isinstance(v, bool):  # If the value is already a boolean, return it directly
        return v
    if v == "True" or v == "true":
        return True
    if v == "False" or v == "false":
        return False


class GradCAM(object):
    """Class for computing Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model):
        # Initialize GradCAM with the specified model
        self.grad_model = model  # Store the model to compute gradients

    def get_gradients_and_outputs(self, inputs, class_idx, use_guided_grads=False):
        # Get gradients and outputs for the target class index
        h_V = (inputs.node_s, inputs.node_v)  # Extract node scalar and vector features
        h_E = (inputs.edge_s, inputs.edge_v)  # Extract edge scalar and vector features
        edge_index = inputs.edge_index  # Extract edge indices
        seq = inputs.seq  # Extract sequence information
        outputs = {}  # Initialize a dictionary to store outputs

        def get_output_hook(name, outputs):
            # Define a hook function to store layer outputs during forward pass
            def hook(module, input, output):
                outputs[name] = output  # Store the output with the specified name

            return hook

        # Register a hook to capture outputs from the target layer
        target_layer = self.grad_model.GVP_encoder.W_out[
            0
        ].scalar_norm  # Select target layer
        target_layer.register_forward_hook(
            get_output_hook("LayerNorm_W_out", outputs)
        )  # Register hook

        # Enable gradients for input features
        h_V = (h_V[0].requires_grad_(), h_V[1].requires_grad_())
        h_E = (h_E[0].requires_grad_(), h_E[1].requires_grad_())

        model_outputs = self.grad_model(
            h_V, edge_index, h_E, seq
        )  # Forward pass through the model

        def extract_column(tensor_tuple, class_idx):
            # Extract the specified column from each tensor in the tuple
            columns = []
            for tensor in tensor_tuple:  # Iterate through each tensor in the tuple
                selected_column = tensor[:, class_idx]  # Select the target class column
                columns.append(
                    selected_column
                )  # Append the selected column to the list
            return torch.stack(columns)  # Stack columns into a tensor

        loss = extract_column(
            model_outputs, class_idx
        ).sum()  # Compute loss for the target class by summing outputs

        self.grad_model.zero_grad()  # Zero gradients before backward pass
        loss.backward(retain_graph=True)  # Backward pass to compute gradients
        grads = outputs[
            "LayerNorm_W_out"
        ].grad  # Extract gradients from the stored outputs

        if use_guided_grads:
            # Apply guided backpropagation by zeroing out negative gradients
            grads = (
                torch.clamp(outputs["LayerNorm_W_out"], min=0.0)
                * torch.clamp(grads, min=0.0)
                * grads
            )

        return outputs, grads  # Return the stored outputs and computed gradients

    def compute_cam(self, output, grad):
        # Compute the class activation map (CAM) using outputs and gradients
        weights = torch.mean(
            grad, dim=(2, 3), keepdim=True
        )  # Compute mean of gradients across spatial dimensions
        cam = torch.sum(
            weights * output, dim=1
        )  # Compute weighted sum of outputs to get CAM
        return cam  # Return the computed CAM

    def heatmap(self, inputs, class_idx, use_guided_grads=False):
        # Generate a heatmap for the specified class index
        output, grad = self.get_gradients_and_outputs(
            inputs, class_idx, use_guided_grads=use_guided_grads
        )  # Get outputs and gradients
        cam = self.compute_cam(output, grad)  # Compute CAM from outputs and gradients
        heatmap = (cam - cam.min()) / (
            cam.max() - cam.min()
        )  # Normalize CAM to get heatmap
        return heatmap.reshape(-1)  # Flatten the heatmap and return


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """

    def __init__(self, model_prefix, gcn=True):
        self.model_prefix = model_prefix

    def predict(self, predict_fasta, task, pooling, contrast, device, args):
        # Predict GO/EC terms and compute CAMs using the specified model and inputs
        output_dim_dict = {
            "bp": 1943,
            "mf": 489,
            "cc": 320,
        }  # Define output dimensions for each task
        config = get_config()  # Load model configuration
        config.pooling = pooling  # Set pooling configuration
        config.contrast = contrast  # Set contrastive learning configuration
        if (
            torch.cuda.is_available() and device != ""
        ):  # Check if CUDA is available and a device is specified
            config.device = "cuda:" + device  # Set the device to the specified GPU
            print(config.device)
        else:
            print(f"unavailable")
        print(
            "### Computing predictions on a single protein..."
        )  # Print status message
        pdb_id, test_fasta_data = extract_single_protein(
            predict_fasta
        )  # Extract protein data from the FASTA file

        # Load GO annotations
        prot_list, prot2annot, goterms, gonames, counts = load_GO_annot_prog(
            "../data/nrPDB-GO_annot.tsv"
        )

        # Set GO term and name arrays, threshold for predictions
        self.gonames = np.asarray(goterms[task])  # Set GO term names for the task
        self.goterms = np.asarray(gonames[task])  # Set GO terms for the task
        self.thresh = 0.1 * np.ones(
            len(self.goterms)
        )  # Set a threshold for GO term predictions
        self.task = task  # Store the task type

        # Create a dataset and dataloader for the input protein data
        test_data = ProteinGraphDataset(
            test_fasta_data, range(len(test_fasta_data)), args, task_list=task
        )  # Create dataset
        val_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            prefetch_factor=2,
        )  # Create DataLoader

        output_dim = output_dim_dict[task]  # Get output dimension for the task

        # Load the model and move it to the specified device
        self.model = GGN_GO(
            2324, 512, 3, 0.95, 0.2, output_dim, pooling, pertub=config.contrast
        ).to(config.device)

        self.model.load_state_dict(torch.load(self.model_prefix))
        self.Y_hat = np.zeros((1, output_dim_dict[task]), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = pdb_id
        self.model.eval()
        loss_function = nn.BCELoss(reduction="none")
        Y_pred = []
        Y_true = []
        proteins = []
        proteins_threshold = []
        threshold = 0.5

        # Iterate over the test dataset
        with torch.no_grad():  # Disable gradient computation for evaluation
            for idx_batch, batch in enumerate(
                val_loader
            ):  # Iterate through the batches
                time.sleep(0.03)  # Pause briefly to avoid excessive speed
                batch = batch.to(config.device)  # Move batch to the specified device
                h_V = (
                    batch.node_s,
                    batch.node_v,
                )  # Extract node scalar and vector features
                h_E = (
                    batch.edge_s,
                    batch.edge_v,
                )  # Extract edge scalar and vector features

                y_true = batch.y  # Get true labels
                if config.contrast:  # If using contrastive learning
                    y_pred, _, _ = self.model(
                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                    )  # Forward pass with contrastive setting
                else:
                    y_pred = self.model(
                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                    )  # Standard forward pass
                y_pred = y_pred.reshape(-1).float()  # Flatten predictions
                losses = loss_function(y_pred, y_true)  # Compute loss for predictions
                total_loss = losses.sum()  # Compute total loss
                average_loss = losses.mean()  # Compute average loss

                if (
                    average_loss > threshold
                ):  # Check if average loss exceeds the threshold
                    print(
                        f"{batch.name}, Total Loss: {total_loss}, Average Loss: {average_loss}"
                    )  # Print loss details
                    proteins_threshold.append(
                        batch.name[0]
                    )  # Add protein name to the list of proteins exceeding the threshold

                proteins.append(batch.name)  # Add protein name to the list
                Y_true.append(
                    batch.y.reshape(1, output_dim).cpu()
                )  # Add true labels to the list
                Y_pred.append(
                    y_pred.reshape(1, output_dim).cpu()
                )  # Add predictions to the list
                self.data[pdb_id] = [
                    batch,
                    batch.seq,
                ]  # Store the batch and sequence information in the data dictionary

            self.Y_hat[0] = Y_pred[0]  # Store predictions in the Y_hat array
            self.prot2goterms[pdb_id] = (
                []
            )  # Initialize the list for storing predicted GO terms for the current protein
            Y_pred_np = Y_pred[0].numpy()  # Convert predictions to a NumPy array
            go_idx = np.where((Y_pred_np[0] >= self.thresh) == True)[
                0
            ]  # Find indices of predictions above the threshold
            for idx in go_idx:  # Iterate over these indices
                if (
                    idx not in self.goidx2chains
                ):  # If the index is not in the GO-to-chains mapping
                    self.goidx2chains[idx] = (
                        set()
                    )  # Initialize a new set for this index
                self.goidx2chains[idx].add(
                    pdb_id
                )  # Add the current protein ID to the set for this GO index
                self.prot2goterms[pdb_id].append(
                    (self.goterms[idx], self.gonames[idx], float(Y_pred_np[0][idx]))
                )  # Store the GO term, name, and prediction score

    def save_predictions(self, output_fn):
        # Save predictions to a JSON file
        print("### Saving predictions to *.json file...")
        # pickle.dump({'pdb_chains': self.test_prot_list, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))
        with open(output_fn, "w") as fw:
            out_data = {
                "pdb_chains": self.test_prot_list,
                "Y_hat": self.Y_hat.tolist(),
                "goterms": self.goterms.tolist(),
                "gonames": self.gonames.tolist(),
            }  # Create a dictionary with prediction data
            json.dump(
                out_data, fw, indent=1
            )  # Write the dictionary to the JSON file with indentation

    def export_csv(self, output_fn, verbose):
        # Export predictions to a CSV file
        with open(output_fn, "w") as csvFile:  # Open the output file in write mode
            writer = csv.writer(
                csvFile, delimiter=",", quotechar='"'
            )  # Create a CSV writer object
            writer.writerow(["### Predictions made by DeepFRI."])  # Write a header row
            writer.writerow(
                ["Protein", "GO_term/EC_number", "Score", "GO_term/EC_number name"]
            )  # Write column names
            if verbose:  # If verbose mode is enabled
                print(
                    "Protein", "GO-term/EC-number", "Score", "GO-term/EC-number name"
                )  # Print the column names
            for (
                prot
            ) in (
                self.prot2goterms
            ):  # Iterate over each protein in the predicted GO terms
                sorted_rows = sorted(
                    self.prot2goterms[prot], key=lambda x: x[2], reverse=True
                )  # Sort predictions by score in descending order
                for row in sorted_rows:  # Iterate over sorted predictions
                    if verbose:  # If verbose mode is enabled
                        print(
                            prot, row[0], "{:.5f}".format(row[2]), row[1]
                        )  # Print the prediction details
                    writer.writerow(
                        [prot, row[0], "{:.5f}".format(row[2]), row[1]]
                    )  # Write the prediction details to the CSV file
        csvFile.close()  # Close the CSV file

    def compute_GradCAM(self, use_guided_grads=False):
        # Compute GradCAM for each function of every predicted protein
        print(
            "### Computing GradCAM for each function of every predicted protein..."
        )  # Print status message
        gradcam = GradCAM(self.model)  # Initialize GradCAM with the loaded model

        self.pdb2cam = {}  # Initialize a dictionary to store CAMs for each protein
        for go_indx in self.goidx2chains:  # Iterate over GO indices
            pred_chains = list(
                self.goidx2chains[go_indx]
            )  # Get the list of predicted protein chains for this GO index
            print(
                "### Computing gradCAM for ",
                self.gonames[go_indx],
                "... [# proteins=",
                len(pred_chains),
                "]",
            )  # Print progress message
            for chain in pred_chains:  # Iterate over each predicted protein chain
                if (
                    chain not in self.pdb2cam
                ):  # If the chain is not already in the CAM dictionary
                    self.pdb2cam[chain] = {}  # Initialize a new entry for this chain
                    self.pdb2cam[chain]["GO_ids"] = []  # Initialize list for GO IDs
                    self.pdb2cam[chain]["GO_names"] = []  # Initialize list for GO names
                    self.pdb2cam[chain][
                        "sequence"
                    ] = None  # Initialize sequence storage
                    self.pdb2cam[chain][
                        "saliency_maps"
                    ] = []  # Initialize list for saliency maps
                self.pdb2cam[chain]["GO_ids"].append(
                    self.goterms[go_indx]
                )  # Add the GO ID to the list
                self.pdb2cam[chain]["GO_names"].append(
                    self.gonames[go_indx]
                )  # Add the GO name to the list
                self.pdb2cam[chain]["sequence"] = self.data[chain][
                    1
                ]  # Store the sequence for the chain
                self.pdb2cam[chain]["saliency_maps"].append(
                    gradcam.heatmap(
                        self.data[chain][0], go_indx, use_guided_grads=use_guided_grads
                    ).tolist()
                )  # Compute and store the saliency map

    def save_GradCAM(self, output_fn):
        # Save computed CAMs to a JSON file
        print("### Saving CAMs to *.json file...")  # Print status message
        with open(output_fn, "w") as fw:  # Open the output file in write mode
            json.dump(
                self.pdb2cam, fw, indent=1
            )  # Write the CAM data to the JSON file with indentation
