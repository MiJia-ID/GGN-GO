import time
from model import *
from nt_xent import NT_Xent
import torch
from torch.autograd import Variable
from sklearn import metrics
from utils import (
    log,
    extract_dataset,
    calculate_filtered_macro_aupr,
    get_arg_train,
    get_arg_val,
)
import argparse
from config import get_config
import numpy as np
from myData import ProteinGraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import math
import random

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

args_train_data = get_arg_train()  # Get arguments for training data
args_val_data = get_arg_val()  # Get arguments for validation data

train_fasta_data = extract_dataset(
    args_train_data.fasta_file
)  # Extract training data from FASTA file
val_fasta_data = extract_dataset(
    args_val_data.fasta_file
)  # Extract validation data from FASTA file

print("train_fasta_data len:", len(train_fasta_data))
print("val_fasta_data len:", len(val_fasta_data))

output_dim_dict = {"bp": 1943, "mf": 489, "cc": 320}


def train(config, task, suffix):
    # Function to train the model with specified configuration, task, and suffix
    train_data = ProteinGraphDataset(
        train_fasta_data, range(len(train_fasta_data)), args_train_data, task_list=task
    )  # Create training dataset
    val_data = ProteinGraphDataset(
        val_fasta_data, range(len(val_fasta_data)), args_val_data, task_list=task
    )  # Create validation dataset

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=args_train_data.num_workers,
        prefetch_factor=1,
    )  # Create DataLoader for training data
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=args_val_data.num_workers,
        prefetch_factor=1,
    )  # Create DataLoader for validation data

    output_dim = output_dim_dict[task]  # Get output dimension for the specified task

    # Initialize the model with specified dimensions and configurations
    model = GGN_GO(
        2324, 256, 3, 0.95, 0.1, output_dim, config.pooling, pertub=config.contrast
    ).to(config.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer,
    )  # Initialize the Adam optimizer with model parameters and configuration

    bce_loss = torch.nn.BCELoss(
        reduce=False
    )  # Define binary cross-entropy loss without reduction

    y_true_all = []  # Initialize list to store all true labels for validation
    for idx_batch, batch in enumerate(val_loader):  # Iterate through validation data
        y_true_all.append(batch.y)  # Append true labels to list
    y_true_all = (
        torch.cat(y_true_all, dim=0).reshape(-1).to(config.device)
    )  # Concatenate and reshape true labels
    y_true_all_aupr = y_true_all.reshape(
        -1, output_dim
    )  # Reshape true labels for AUPR calculation

    train_loss = []
    val_aupr = []
    es = 0

    for ith_epoch in range(config.max_epochs):  # Iterate through epochs
        for idx_batch, batch in enumerate(
            train_loader
        ):  # Iterate through training data
            batch = batch.to(config.device)  # Move batch to the specified device

            model.train()

            h_V = (
                batch.node_s,
                batch.node_v,
            )  # Extract node scalar and vector features
            h_E = (
                batch.edge_s,
                batch.edge_v,
            )  # Extract edge scalar and vector features

            if config.contrast:  # If contrastive learning is enabled
                y_pred, g_feat1, g_feat2 = model(
                    h_V, batch.edge_index, h_E, batch.seq, batch.batch
                )  # Forward pass with contrastive learning
                y_true = batch.y.reshape(-1)  # Reshape true labels
                y_pred = y_pred.reshape(-1).float()  # Reshape predictions

                _loss = bce_loss(y_pred, y_true)  # Compute binary cross-entropy loss
                _loss = _loss.mean()  # Compute mean loss

                criterion = NT_Xent(g_feat1.shape[0], 0.1, 1)  # Initialize NT-Xent loss
                cl_loss = 0.05 * criterion(
                    g_feat1, g_feat2
                )  # Compute contrastive loss with scaling factor

                loss = (
                    _loss + cl_loss
                )  # Combine binary cross-entropy loss and contrastive loss
            else:
                y_pred = model(
                    h_V, batch.edge_index, h_E, batch.seq, batch.batch
                )  # Standard forward pass

                y_true = batch.y.to(
                    config.device
                )  # Move true labels to the specified device
                y_pred = y_pred.reshape([-1]).float()  # Reshape predictions
                y_true = y_true.float()  # Convert true labels to float

                loss = bce_loss(y_pred, y_true)  # Compute binary cross-entropy loss
                loss = loss.mean()

            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)}")
            train_loss.append(
                loss.clone().detach().cpu().numpy()
            )  # Append loss to training loss list
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

        model.eval()
        y_pred_all = []
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                time.sleep(0.03)
                batch = batch.to(config.device)
                if config.contrast:
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)

                    y_pred, _, _ = model(
                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                    )  # Forward pass with contrastive learning
                else:
                    y_pred = model(batch.to(config.device))  # Standard forward pass
                y_pred_all.append(y_pred)  # Append predictions to list

                log(f"{idx_batch}/{ith_epoch} val_epoch")

            y_pred_all = (
                torch.cat(y_pred_all, dim=0).reshape(-1).to(config.device)
            )  # Concatenate and reshape predictions
            eval_loss = bce_loss(
                y_pred_all, y_true_all
            ).mean()  # Compute mean binary cross-entropy loss for validation

            y_pred_all = y_pred_all.reshape(
                -1, output_dim
            )  # Reshape predictions for AUPR calculation

            aupr = calculate_filtered_macro_aupr(
                y_true_all_aupr.cpu().numpy(), y_pred_all.cpu().numpy()
            )  # Calculate AUPR
            val_aupr.append(aupr)  # Append AUPR score to validation AUPR list
            log(
                f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss), 3)} ||| aupr: {round(float(aupr), 3)} "
            )  # Log validation loss and AUPR

            if ith_epoch == 0:
                best_eval_loss = eval_loss
            if best_eval_loss < eval_loss:
                best_eval_loss = eval_loss
                es = 0
                torch.save(model.state_dict(), config.model_save_path + task + f".pt")
            else:
                es += 1
                print("Counter {} of 5".format(es))
                torch.save(model.state_dict(), config.model_save_path + task + f"es.pt")
            if es > 5:
                torch.save(
                    {
                        "train_bce": train_loss,  # Save training loss history
                        "val_bce": eval_loss,  # Save validation loss
                        "val_aupr": val_aupr,  # Save validation AUPR scores
                    },
                    config.loss_save_path + task + f"_val.pt",
                )  # Save loss and AUPR data
                break


def str2bool(v):
    """Convert a string to a boolean value."""
    if isinstance(v, bool):
        return v
    if v == "True" or v == "true":
        return True
    if v == "False" or v == "false":
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", type=str, default="bp", choices=["bp", "mf", "cc"], help=""
    )
    p.add_argument("--suffix", type=str, default="False", help="")
    p.add_argument("--device", type=str, default="1", help="")
    p.add_argument(
        "--pooling",
        default="MTP",
        type=str,
        choices=["MTP", "GMP"],
        help="Multi-set transformer pooling or Global max pooling",
    )
    p.add_argument(
        "--contrast",
        default=True,
        type=str2bool,
        help="whether to do contrastive learning",
    )
    p.add_argument(
        "--AF2model",
        default=False,
        type=str2bool,
        help="whether to use AF2model for training",
    )
    p.add_argument("--batch_size", type=int, default=2, help="")

    args = p.parse_args()
    config = get_config()
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if torch.cuda.is_available() and args.device != "":
        config.device = "cuda:" + args.device
        print(config.device)
    else:
        print(f"unavailable")
    print(args)
    config.pooling = args.pooling
    config.contrast = args.contrast
    config.AF2model = args.AF2model
    train(config, args.task, args.suffix)
