import os
import csv
import glob
import json

import time
from utils import extract_sequences_from_fasta,save_sequences_to_fasta
from model import *
import torch
from utils import load_GO_annot_prog,log,extract_single_protein
import argparse
from config import get_config
from myData import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import warnings

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False



class GradCAM(object):

    def __init__(self,model):
        self.grad_model = model


    def get_gradients_and_outputs(self, inputs, class_idx, use_guided_grads=False):
        h_V = (inputs.node_s, inputs.node_v)
        h_E = (inputs.edge_s, inputs.edge_v)
        edge_index = inputs.edge_index
        seq= inputs.seq
        outputs = {}

        def get_output_hook(name,outputs):
            def hook(module, input, output):
                outputs[name] = output

            return hook

        target_layer = self.grad_model.GVP_encoder.W_out[0].scalar_norm
        target_layer.register_forward_hook(get_output_hook('LayerNorm_W_out', outputs))

        h_V = (h_V[0].requires_grad_(), h_V[1].requires_grad_())
        h_E = (h_E[0].requires_grad_(), h_E[1].requires_grad_())

        model_outputs = self.grad_model(h_V, edge_index, h_E, seq)

        def extract_column(tensor_tuple, class_idx):
            columns = []
            for tensor in tensor_tuple:
                selected_column = tensor[:, class_idx]
                columns.append(selected_column)
            return torch.stack(columns)

        loss = extract_column(model_outputs, class_idx).sum()

        self.grad_model.zero_grad()
        loss.backward(retain_graph=True)
        grads = outputs['LayerNorm_W_out'].grad

        if use_guided_grads:
            grads = torch.clamp(outputs['LayerNorm_W_out'], min=0.0) * torch.clamp(grads, min=0.0) * grads

        return outputs, grads

    def compute_cam(self,output, grad):

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * output, dim=1)
        return cam

    def heatmap(self, inputs, class_idx, use_guided_grads=False):
        output, grad = self.get_gradients_and_outputs(inputs,class_idx, use_guided_grads=use_guided_grads)
        cam = self.compute_cam(output, grad)
        heatmap = (cam - cam.min())/(cam.max() - cam.min())

        return heatmap.reshape(-1)


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, gcn=True):
        self.model_prefix = model_prefix


    def predict(self, predict_fasta, task, pooling, contrast, device, args):
        output_dim_dict = {'bp': 1943, 'mf': 489, 'cc': 320}
        config = get_config()
        config.pooling = pooling
        config.contrast = contrast
        if torch.cuda.is_available() and device != '':
            config.device = "cuda:" + device
            print(config.device)
        else:
            print(f"unavailable")
        print ("### Computing predictions on a single protein...")
        pdb_id,test_fasta_data = extract_single_protein(predict_fasta)

        prot_list, prot2annot, goterms, gonames, counts = load_GO_annot_prog(
            '../data/nrPDB-GO_annot.tsv')

        self.gonames = np.asarray(goterms[task])
        self.goterms = np.asarray(gonames[task])
        self.thresh = 0.1 * np.ones(len(self.goterms))
        self.task = task

        test_data = ProteinGraphDataset(test_fasta_data, range(len(test_fasta_data)), args, task_list=task)
        val_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False,
                                num_workers=args.num_workers, prefetch_factor=2)

        output_dim = output_dim_dict[task]

        self.model = GLMSite(2324, 512, 3, 0.95, 0.2, output_dim, pooling, pertub=config.contrast).to(config.device)

        self.model.load_state_dict(torch.load(self.model_prefix))
        self.Y_hat = np.zeros((1, output_dim_dict[task]), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = pdb_id
        self.model.eval()
        loss_function = nn.BCELoss(reduction='none')
        Y_pred = []
        Y_true = []
        proteins = []
        proteins_threshold = []
        threshold = 0.5

        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                time.sleep(0.03)
                batch = batch.to(config.device)
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)

                y_true = batch.y
                if config.contrast:
                    y_pred, _, _ = self.model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                else:
                    y_pred = self.model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                y_pred = y_pred.reshape(-1).float()
                losses = loss_function(y_pred, y_true)
                total_loss = losses.sum()
                average_loss = losses.mean()

                if average_loss > threshold:
                    print(f"{batch.name}, 总损失: {total_loss}, 平均损失: {average_loss}")
                    proteins_threshold.append(batch.name[0])

                proteins.append(batch.name)
                Y_true.append(batch.y.reshape(1, output_dim).cpu())
                Y_pred.append(y_pred.reshape(1, output_dim).cpu())
                self.data[pdb_id] = [batch, batch.seq]

            self.Y_hat[0] = Y_pred[0]
            self.prot2goterms[pdb_id] = []
            Y_pred_np = Y_pred[0].numpy()
            go_idx = np.where((Y_pred_np[0] >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(pdb_id)
                self.prot2goterms[pdb_id].append((self.goterms[idx], self.gonames[idx], float(Y_pred_np[0][idx])))


    def save_predictions(self, output_fn):
        print ("### Saving predictions to *.json file...")
        # pickle.dump({'pdb_chains': self.test_prot_list, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'Y_hat': self.Y_hat.tolist(),
                        'goterms': self.goterms.tolist(),
                        'gonames': self.gonames.tolist()}
            json.dump(out_data, fw, indent=1)

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print ('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print (prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
        csvFile.close()




    def compute_GradCAM(self, use_guided_grads=False):
        print ("### Computing GradCAM for each function of every predicted protein...")
        gradcam = GradCAM(self.model)

        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = self.data[chain][1]
                self.pdb2cam[chain]['saliency_maps'].append(gradcam.heatmap(self.data[chain][0],go_indx, use_guided_grads=use_guided_grads).tolist())

    def save_GradCAM(self, output_fn):
        print ("### Saving CAMs to *.json file...")
        with open(output_fn, 'w') as fw:
            json.dump(self.pdb2cam, fw, indent=1)

