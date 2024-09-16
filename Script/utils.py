import torch
import argparse
import numpy as np
from torch_geometric.data import Data, Dataset, Batch
import csv
import glob
import os
import shutil
import numpy as np
from joblib import Parallel, delayed, cpu_count
import sys
from datetime import datetime
from tqdm import tqdm
from Bio.PDB import PDBList
from Bio import PDB
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn import metrics
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from sklearn.utils import resample
import urllib.request
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from sklearn.metrics import precision_recall_curve, auc
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


RES2ID = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "-": 20,
}


def aa2idx(seq):
    # Convert amino acid sequence letters into numerical indices
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype="|S1").view(
        np.uint8
    )  # Convert the alphabet to byte view
    idx = np.array(list(seq), dtype="|S1").view(
        np.uint8
    )  # Convert the sequence to byte view
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx


def protein_graph(sequence, edge_index, esm_embed):
    # Create a protein graph with sequence, edge indices, and embeddings
    seq_code = aa2idx(sequence)  # Convert sequence to numerical indices
    seq_code = torch.IntTensor(seq_code)  # Convert to PyTorch tensor
    edge_index = torch.LongTensor(edge_index)  # Convert edge indices to tensor
    data = Data(
        x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code
    )  # Create Data object with features and edges
    return data


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """
    Parallel map using joblib.
    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.
    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(*d, **kwargs)
        for i, d in tqdm(enumerate(data), desc=desc)
    )

    return results


def pmap_single(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """
    Parallel map using joblib.
    Parameters
    ----------
    pickleable_fn : callable
      Function to map over data.
    data : iterable
      Data over which we want to parallelize the function call.
    n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
    verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
      Additional arguments for :attr:`pickleable_fn`.
    Returns
    -------
    list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(d, **kwargs) for i, d in tqdm(enumerate(data), desc=desc)
    )

    return results


def retrieve_pdb(pdb, pdir):
    # Retrieve a PDB file from the RCSB PDB database
    pdb_url_base = "http://www.rcsb.org/pdb/files/"  # Base URL for PDB files
    url = f"{pdb_url_base}{pdb}.pdb"
    print(url)

    if not os.path.exists(pdir):
        os.makedirs(pdir)

    fname = f"{pdir}{pdb}.pdb"
    try:
        with urllib.request.urlopen(url) as f:

            with open(fname, "wb") as g:
                g.write(f.read())
    except urllib.error.HTTPError as e:
        print(f"Failed to retrieve PDB {pdb}. HTTPError: {e}")
        os.remove(fname)
        return


def load_predicted_PDB(pdbfile):
    # Load a PDB file and generate a C_alpha distance matrix
    parser = PDBParser()  # Initialize PDB parser
    structure = parser.get_structure(
        pdbfile.split("/")[-1].split(".")[0], pdbfile
    )  # Parse the structure from the PDB file
    residues = [
        r for r in structure.get_residues()
    ]  # Extract residues from the structure

    # Extract sequence from atom lines in the PDB file
    records = SeqIO.parse(pdbfile, "pdb-atom")
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))  # Initialize distance matrix
    for x in range(len(residues)):  # Compute pairwise distances between residues
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()  # Get coordinates of alpha carbon
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)  # Compute Euclidean distance

    return distances, seqs[0]


def load_FASTA(filename):
    # Load sequences from a FASTA file
    infile = open(filename, "rU")  # Open FASTA file
    entries = []  # Initialize list for sequences
    proteins = []  # Initialize list for protein IDs
    for entry in SeqIO.parse(infile, "fasta"):  # Parse sequences from FASTA
        entries.append(str(entry.seq))  # Add sequence to list
        proteins.append(str(entry.id))  # Add protein ID to list
    return proteins, entries  # Return protein IDs and sequences


def load_GO_annot_prog(filename):
    # Load GO annotations from a file
    onts = ["mf", "bp", "cc"]
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode="r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # Read molecular function terms and names
        next(reader, None)  # Skip the headers
        goterms[onts[0]] = next(reader)  # Read GO terms
        next(reader, None)  # Skip the headers
        gonames[onts[0]] = next(reader)  # Read GO names

        # Read biological process terms and names
        next(reader, None)  # Skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # Skip the headers
        gonames[onts[1]] = next(reader)

        # Read cellular component terms and names
        next(reader, None)  # Skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        prot_res = []  # Initialize list for protein results
        for row in reader:  # Iterate over rows in the annotation file
            prot, prot_goterms = row[0], row[1:]  # Extract protein ID and GO terms
            prot_res.append(prot)  # Add protein to the list
            prot2annot[prot] = {
                ont: [] for ont in onts
            }  # Initialize dictionary for each protein
            for i in range(3):  # Iterate over ontologies
                goterm_indices = [
                    goterms[onts[i]].index(goterm)
                    for goterm in prot_goterms[i].split(",")
                    if goterm != ""
                ]
                prot2annot[prot][onts[i]] = np.zeros(
                    len(goterms[onts[i]])
                )  # Initialize array for GO term presence
                prot2annot[prot][onts[i]][
                    goterm_indices
                ] = 1.0  # Set presence to 1.0 for identified GO terms
                counts[onts[i]][goterm_indices] += 1.0  # Increment counts for GO terms
    return (
        prot_res,
        prot2annot,
        goterms,
        gonames,
        counts,
    )  # Return protein annotations and related data


def log(*args):
    print(f"[{datetime.now()}]", *args)


def PR_metrics(y_true, y_pred):
    # Compute precision-recall metrics for different thresholds
    precision_list = []
    recall_list = []
    threshold = np.arange(0.01, 1.01, 0.01)
    for T in threshold:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_list.append(
                metrics.precision_score(y_true, np.where(y_pred >= T, 1, 0))
            )  # Compute precision
            recall_list.append(
                metrics.recall_score(y_true, np.where(y_pred >= T, 1, 0))
            )  # Compute recall
    return np.array(precision_list), np.array(
        recall_list
    )  # Return precision and recall arrays


def fmax(y_true, y_pred_probs):
    # Compute F-max score for precision-recall
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_probs
    )  # Compute precision-recall curve
    f1_scores = 2 * (precision * recall) / (precision + recall)  # Compute F1 scores
    f1_scores = f1_scores[~np.isnan(f1_scores)]  # Remove NaN values
    f_max = np.max(f1_scores)  # Find maximum F1 score

    return f_max


def smin(y_true, y_scores):
    # Compute S-min score for precision-recall
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores
    )  # Compute precision-recall curve
    thresholds = np.append(thresholds, 1)  # Append 1 to thresholds
    Smin = float("inf")  # Initialize S-min with infinity

    for t in thresholds:  # Iterate over thresholds
        y_pred = [
            1 if score >= t else 0 for score in y_scores
        ]  # Generate predictions based on threshold

        TP = sum(
            1 for true, pred in zip(y_true, y_pred) if true == pred == 1
        )  # Compute true positives
        FP = sum(
            1 for true, pred in zip(y_true, y_pred) if true != pred and pred == 1
        )  # Compute false positives
        FN = sum(
            1 for true, pred in zip(y_true, y_pred) if true != pred and true == 1
        )  # Compute false negatives

        if (
            TP + FP > 0 and FN > 0
        ):  # If there are both true positives and false negatives
            IC_TP = -np.log2(
                TP / (TP + FP)
            )  # Compute information content for true positives
            IC_FN = -np.log2(
                FN / (FN + TP)
            )  # Compute information content for false negatives
        else:
            IC_TP = IC_FN = (
                0  # Set information content to zero if no positives or negatives
            )

        ru = IC_FN  # Set ru as information content of false negatives
        mi = IC_TP  # Set mi as information content of true positives

        S = np.sqrt(ru**2 + mi**2)  # Compute S score

        Smin = min(Smin, S)  # Update S-min with the minimum S score

    return Smin  # Return S-min score


# Mapping from single-letter to three-letter amino acid codes
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# Reverse mapping from three-letter to single-letter amino acid codes
restype_3to1 = {v: k for k, v in restype_1to3.items()}


def extract_protein_info(protein_name, pdb_file_path="../data/pdb_dir/"):
    # Extract sequence and atom coordinates from a PDB file
    pdb_file_name = protein_name + ".pdb"  # Construct PDB file name
    pdb = os.path.join(pdb_file_path, pdb_file_name)  # Construct full path to PDB file

    parser = PDBParser()
    struct = parser.get_structure("x", pdb)
    model = struct[0]
    chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]
    sequence = ""
    atom_coords = []

    for residue in chain:  # Iterate over residues in the chain
        try:
            coords = np.array(
                [
                    residue["N"].get_coord(),
                    residue["CA"].get_coord(),
                    residue["C"].get_coord(),
                    residue["O"].get_coord(),
                ]
            ).astype(
                np.float32
            )  # Get coordinates of backbone atoms
            atom_coords.append(coords)  # Append coordinates to the list
            residue_name = residue.get_resname()  # Get residue name

            if (
                residue_name in restype_3to1
            ):  # Check if residue name is in the dictionary
                sequence += restype_3to1[
                    residue_name
                ]  # Convert to single-letter code and add to sequence
            else:
                sequence += "X"  # Add 'X' for unknown residues
        except KeyError as e:  # Handle missing atoms
            print(protein_name)
            print(f"Missing atom in residue: {residue.id}, error: {e}")
            continue

    np.save(
        "../data/bpDataSet/structure_data/" + protein_name + ".npy",
        np.array(atom_coords),
    )  # Save coordinates to file

    return sequence  # Return the sequence


def rename_files_to_uppercase(folder_path="../data/pdb_dir"):
    """
    Rename all .pdb files in the specified folder to uppercase names.
    """
    if not os.path.isdir(folder_path):  # Check if the folder path is valid
        print("The specified path does not exist or is not a folder.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):
            name_part, extension = os.path.splitext(filename)
            new_name = name_part.upper() + extension
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)
            shutil.move(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_name}"')

    print("All files have been renamed.")


def extract_dataset(fasta_file):
    # Extract sequences from a FASTA file into a dictionary
    train_fasta_data = {}

    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split(">")[1].replace("\n", "")
            seq = fasta_ori[i + 1].replace("\n", "")
            train_fasta_data[name] = seq.replace("\n", "")
    return train_fasta_data


def extract_single_protein(fasta_file):
    # Extract a single protein's sequence from a FASTA file
    train_fasta_data = {}

    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split(">")[1].replace("\n", "")
            seq = fasta_ori[i + 1].replace("\n", "")
            train_fasta_data[name] = seq.replace("\n", "")
    return name, train_fasta_data


def calculate_filtered_macro_aupr(y_true_all, y_pred_all):
    """
    Calculate filtered macro-average AUPR score.
    """
    print("y_true_all:", y_true_all.shape)
    print("y_pred_all:", y_pred_all.shape)
    keep_goidx = np.where(y_true_all.sum(axis=0) > 0)[0]
    print("### Number of functions =%d" % (len(keep_goidx)))

    y_true_filtered = y_true_all[:, keep_goidx]
    y_pred_filtered = y_pred_all[:, keep_goidx]

    macro_aupr = metrics.average_precision_score(
        y_true_filtered, y_pred_filtered, average="macro"
    )

    return macro_aupr


def get_arg_train():
    parser_train = argparse.ArgumentParser()
    parser_train.add_argument(
        "--fasta_file", type=str, default="../data/nrPDB-GO_train_sequences.fasta"
    )
    parser_train.add_argument(
        "--prottrans_output_path", type=str, default="../data/bpDataSet/prottrans/"
    )
    parser_train.add_argument(
        "--esm2_output_path", type=str, default="../data/bpDataSet/esm2/"
    )
    parser_train.add_argument("--pdb_dir", type=str, default="../data/pdb_dir/")
    parser_train.add_argument(
        "--proc_pdb_output_path", type=str, default="../data/bpDataSet/structure_data/"
    )
    parser_train.add_argument(
        "--dssp_output_path", type=str, default="../data/bpDataSet/dssp/"
    )
    parser_train.add_argument("--num_workers", type=int, default=8)
    args_train_data = parser_train.parse_args()
    return args_train_data


def get_arg_val():
    parser_val = argparse.ArgumentParser()
    parser_val.add_argument(
        "--fasta_file", type=str, default="../data/nrPDB-GO_val_sequences.fasta"
    )
    parser_val.add_argument(
        "--prottrans_output_path", type=str, default="../data/bpDataSet/prottrans/"
    )
    parser_val.add_argument(
        "--esm2_output_path", type=str, default="../data/bpDataSet/esm2/"
    )
    parser_val.add_argument("--pdb_dir", type=str, default="../data/pdb_dir/")
    parser_val.add_argument(
        "--proc_pdb_output_path", type=str, default="../data/bpDataSet/structure_data/"
    )
    parser_val.add_argument(
        "--dssp_output_path", type=str, default="../data/bpDataSet/dssp/"
    )
    parser_val.add_argument("--num_workers", type=int, default=8)
    args_val_data = parser_val.parse_args()
    return args_val_data


def extract_sequences_from_fasta(fasta_file, sequence_names):
    with open(fasta_file, "r") as file:
        sequences = {}
        current_seq_name = None
        for line in file:
            if line.startswith(">"):
                current_seq_name = line[1:].strip().split()[0]
                if current_seq_name in sequence_names:
                    sequences[current_seq_name] = ""
            else:
                if current_seq_name in sequence_names:
                    sequences[current_seq_name] += line.strip()
        return sequences


def save_sequences_to_fasta(sequences, output_file):
    with open(output_file, "w") as file:
        for name, seq in sequences.items():
            file.write(f">{name}\n{seq}\n")


if __name__ == "__main__":
    prot_list, prot2annot, goterms, gonames, counts = load_GO_annot_prog(
        "../data/nrPDB-GO_annot.tsv"
    )
    print(prot2annot)
