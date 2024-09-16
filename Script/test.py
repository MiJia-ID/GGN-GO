import time
from utils import extract_sequences_from_fasta, save_sequences_to_fasta
from model import *
import torch
import pickle
from utils import load_GO_annot_prog, log, extract_dataset
import argparse
from config import get_config
from myData import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fasta_file", type=str, default="../data/nrPDB-GO_test_sequences.fasta"
)
parser.add_argument(
    "--prottrans_output_path", type=str, default="../data/bpDataSet/prottrans/"
)
parser.add_argument("--esm2_output_path", type=str, default="../data/bpDataSet/esm2/")
parser.add_argument("--pdb_dir", type=str, default="../data/pdb_dir/")
parser.add_argument(
    "--proc_pdb_output_path", type=str, default="../data/bpDataSet/structure_data/"
)
parser.add_argument("--dssp_output_path", type=str, default="../data/bpDataSet/dssp/")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pkl_file_path", type=str, default="../Model/bp_model_test.pckl")
args_data = parser.parse_args()

test_fasta_data = extract_dataset(args_data.fasta_file)

output_dim_dict = {"bp": 1943, "mf": 489, "cc": 320}


def test(config, task, suffix):
    prot_list, prot2annot, goterms, gonames, counts = load_GO_annot_prog(
        "../data/nrPDB-GO_annot.tsv"
    )
    goterms = goterms[task]
    gonames = gonames[task]

    test_data = ProteinGraphDataset(
        test_fasta_data, range(len(test_fasta_data)), args_data, task_list=task
    )
    val_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args_data.num_workers,
        prefetch_factor=2,
    )

    output_dim = output_dim_dict[task]

    model = GGN(
        2324, 512, 3, 0.95, 0.2, output_dim, config.pooling, pertub=config.contrast
    ).to(config.device)

    model.load_state_dict(torch.load("../Model/model_" + "bp" + ".pt"))

    model.eval()
    Y_pred = []
    Y_true = []
    proteins = []

    with torch.no_grad():
        for idx_batch, batch in enumerate(val_loader):
            batch = batch.to(config.device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)

            if config.contrast:
                y_pred, _, _ = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
            else:
                y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
            y_pred = y_pred.reshape(-1).float()

            proteins.append(batch.name)
            Y_true.append(batch.y.reshape(1, output_dim).cpu())
            Y_pred.append(y_pred.reshape(1, output_dim).cpu())

        with open(args_data.pkl_file_path, "wb") as file:
            pickle.dump(
                {
                    "proteins": np.asarray(proteins),
                    "Y_pred": np.concatenate(Y_pred, axis=0),
                    "Y_true": np.concatenate(Y_true, axis=0),
                    "ontology": task,
                    "goterms": goterms,
                    "gonames": gonames,
                },
                file,
            )
        return


def str2bool(v):
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
    p.add_argument("--device", type=str, default="3", help="")
    p.add_argument("--esmembed", default=False, type=str2bool, help="")
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
    config.optimizer["lr"] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if torch.cuda.is_available() and args.device != "":
        config.device = "cuda:" + args.device
        print(config.device)
    else:
        print(f"unavailable")
    config.esmembed = args.esmembed
    print(args)
    config.pooling = args.pooling
    config.contrast = args.contrast
    config.AF2model = args.AF2model
    test(config, args.task, args.suffix)
