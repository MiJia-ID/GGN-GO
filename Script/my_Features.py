import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import esm
import pickle


def get_prottrans(fasta_file, output_path):
    os.environ["OMP_NUM_THREADS"] = (
        "1"  # Set the number of OpenMP threads to 1 to avoid multithreading issues
    )
    parser = argparse.ArgumentParser()  # Create an argument parser
    parser.add_argument(
        "--gpu", type=str, default="1"
    )  # Add GPU argument to specify which GPU to use
    args = parser.parse_args()  # Parse the command-line arguments
    gpu = args.gpu  # Get the GPU argument value

    ID_list = []  # Initialize list to store sequence IDs
    seq_list = []  # Initialize list to store sequences
    with open(fasta_file, "r") as f:  # Open the FASTA file for reading
        lines = f.readlines()  # Read all lines from the file
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespaces
        if line and line[0] == ">":  # Check if line is a sequence ID line
            ID_list.append(line[1:])  # Append sequence ID to the list
        elif line:  # If line is a sequence line
            seq_list.append(
                " ".join(list(line))
            )  # Append sequence with spaces between characters

    # Display the first 9 sequences for verification
    for id, seq in zip(ID_list[:9], seq_list[:9]):
        print(f"ID: {id}")
        print(f"Sequence: {seq[:]}...")
        print("len:", len(seq))

    model_path = "./Prot-T5-XL-U50"  # Path to the Prot-T5 model
    tokenizer = T5Tokenizer.from_pretrained(
        model_path, do_lower_case=False
    )  # Load tokenizer without converting text to lowercase
    model = T5EncoderModel.from_pretrained(model_path)  # Load the T5 encoder model
    gc.collect()  # Invoke garbage collector to free up memory

    device = torch.device(
        "cuda:" + gpu if torch.cuda.is_available() and gpu else "cpu"
    )  # Set device to GPU if available, otherwise CPU
    model = model.eval().to(
        device
    )  # Set model to evaluation mode and move to the specified device

    print(
        next(model.parameters()).device
    )  # Print the device where the model's parameters are located
    print("starttime")
    starttime = datetime.datetime.now()  # Record the start time
    print(starttime)
    batch_size = 1  # Set the batch size for processing sequences

    # Process sequences in batches
    for i in tqdm(range(0, len(ID_list), batch_size)):  # Iterate over sequence batches
        batch_ID_list = ID_list[i : i + batch_size]  # Get the current batch of IDs
        batch_seq_list = seq_list[
            i : i + batch_size
        ]  # Get the current batch of sequences

        batch_seq_list = [
            re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list
        ]  # Replace uncommon amino acids with 'X'

        ids = tokenizer.batch_encode_plus(
            batch_seq_list, add_special_tokens=True, padding=True
        )  # Tokenize and pad sequences
        input_ids = torch.tensor(ids["input_ids"]).to(
            device
        )  # Convert input IDs to a tensor and move to device
        attention_mask = torch.tensor(ids["attention_mask"]).to(
            device
        )  # Convert attention mask to a tensor and move to device

        with torch.no_grad():
            embedding = model(
                input_ids=input_ids, attention_mask=attention_mask
            )  # Generate embeddings using the model
        embedding = (
            embedding.last_hidden_state.cpu().numpy()
        )  # Convert embeddings to a numpy array on CPU

        # Save embeddings for each sequence in the batch
        for seq_num in range(len(embedding)):
            seq_len = (
                attention_mask[seq_num] == 1
            ).sum()  # Determine the actual length of the sequence without padding
            seq_emd = embedding[seq_num][
                : seq_len - 1
            ]  # Exclude the last token (special token)

            np.save(
                os.path.join(output_path, batch_ID_list[seq_num]), seq_emd
            )  # Save the sequence embedding as a .npy file

    endtime = datetime.datetime.now()  # Record the end time
    print("endtime")
    print(endtime)


def get_pdb_xyz(pdb_file, ref_seq):
    current_pos = -1000  # Initialize current position to a negative large value
    X = []  # Initialize list to store atomic coordinates
    current_aa = (
        {}
    )  # Initialize dictionary to store current amino acid atoms: 'N', 'CA', 'C', 'O'
    try:
        for line in pdb_file:  # Iterate through each line in the PDB file
            # Check if the line is an ATOM line and marks a new amino acid or is a termination line
            if (
                line[0:4].strip() == "ATOM" and (line[22:26].strip()) != current_pos
            ) or line[0:4].strip() == "TER":
                if current_aa != {}:  # If the current amino acid is not empty
                    X.append(
                        [
                            current_aa["N"],
                            current_aa["CA"],
                            current_aa["C"],
                            current_aa["O"],
                        ]
                    )  # Append the coordinates to X
                    current_aa = {}  # Reset the current amino acid dictionary
                if (
                    line[0:4].strip() != "TER"
                ):  # If not a termination line, update current position
                    current_pos = line[22:26].strip()

            if line[0:4].strip() == "ATOM":  # If the line describes an atom
                atom = line[13:16].strip()  # Extract the atom type
                if atom in [
                    "N",
                    "CA",
                    "C",
                    "O",
                ]:  # If the atom is one of the backbone atoms
                    xyz = np.array(
                        [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
                    ).astype(
                        np.float32
                    )  # Extract coordinates
                    current_aa[atom] = (
                        xyz  # Store the coordinates in the current amino acid dictionary
                    )
    except:
        print("111")
        return None
    if len(X) == len(
        ref_seq
    ):  # Check if the length of the coordinates matches the reference sequence length
        return np.array(X)  # Return the array of coordinates
    else:
        print("len(X)", len(X))
        print("len(ref_seq)", len(ref_seq))
        return None


def proc_pdb(fasta_file, pdb_path, output_path):
    pdbfasta = {}  # Initialize dictionary to map PDB file names to sequences
    with open(fasta_file) as r1:  # Open the FASTA file
        fasta_ori = r1.readlines()  # Read all lines
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":  # If the line is a header line
            name = (
                fasta_ori[i].split(">")[1].replace("\n", "")
            )  # Extract the name of the sequence
            seq = fasta_ori[i + 1].replace("/n", "")  # Extract the sequence
            pdbfasta[name] = seq.replace("\n", "")  # Map the name to the sequence

    for key in pdbfasta.keys():  # Iterate through each PDB name
        with open(
            pdb_path + key + ".pdb", "r"
        ) as r1:  # Open the corresponding PDB file
            pdb_file = r1.readlines()  # Read the PDB file lines
        coord = get_pdb_xyz(
            pdb_file, pdbfasta[key]
        )  # Extract the coordinates using the get_pdb_xyz function
        if coord is None:  # Check if coordinate extraction was unsuccessful
            print("pdb:", key)  # Print the name of the PDB that failed
        np.save(
            output_path + key + ".npy", coord
        )  # Save the coordinates as a .npy file


def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = "./dssp"

    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"  # Define standard amino acids
        SS_type = "HBEGITSC"  # Define secondary structure types
        rASA_std = [
            115,
            135,
            150,
            190,
            210,
            75,
            195,
            175,
            200,
            170,
            185,
            160,
            145,
            180,
            225,
            115,
            140,
            155,
            255,
            230,
        ]  # Standard relative accessible surface areas

        with open(dssp_file, "r") as f:  # Open DSSP output file
            lines = f.readlines()  # Read all lines

        seq = ""  # Initialize sequence string
        dssp_feature = []  # Initialize list to store DSSP features

        p = 0
        while lines[p].strip()[0] != "#":  # Find the line starting with DSSP data
            p += 1
        for i in range(p + 1, len(lines)):  # Iterate over DSSP data lines
            aa = lines[i][13]  # Extract amino acid type
            if aa == "!" or aa == "*":  # Skip unknown residues
                continue
            seq += aa  # Append amino acid to sequence
            SS = lines[i][16]  # Extract secondary structure type
            if SS == " ":
                SS = "C"  # Default to coil if no secondary structure is specified
            SS_vec = np.zeros(
                9
            )  # Initialize a vector for secondary structure one-hot encoding
            SS_vec[SS_type.find(SS)] = 1  # Set the appropriate position to 1
            PHI = float(lines[i][103:109].strip())  # Extract phi angle
            PSI = float(lines[i][109:115].strip())  # Extract psi angle
            ACC = float(lines[i][34:38].strip())  # Extract accessible surface area
            ASA = (
                min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            )  # Normalize the ASA value
            dssp_feature.append(
                np.concatenate((np.array([PHI, PSI, ASA]), SS_vec))
            )  # Append DSSP features

        return seq, dssp_feature  # Return sequence and DSSP features

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(
            ref_seq, seq
        )  # Align reference sequence and DSSP sequence
        ref_seq = alignments[0].seqA  # Get aligned reference sequence
        seq = alignments[0].seqB  # Get aligned DSSP sequence

        SS_vec = np.zeros(9)  # Define a vector for missing secondary structures
        SS_vec[-1] = 1
        padded_item = np.concatenate(
            (np.array([360, 360, 0]), SS_vec)
        )  # Define padding for gaps

        new_dssp = []  # Initialize list for padded DSSP features
        for aa in seq:
            if aa == "-":  # If gap in DSSP sequence
                new_dssp.append(padded_item)  # Add padding
            else:
                new_dssp.append(dssp.pop(0))  # Otherwise, add DSSP feature

        matched_dssp = []  # Initialize list for matched DSSP features
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":  # Skip gaps in reference sequence
                continue
            matched_dssp.append(new_dssp[i])  # Add matched DSSP feature

        return matched_dssp  # Return matched DSSP features

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)  # Convert DSSP features to NumPy array
        angle = dssp_feature[:, 0:2]  # Extract phi and psi angles
        ASA_SS = dssp_feature[:, 2:]  # Extract ASA and secondary structure vectors

        radian = angle * (np.pi / 180)  # Convert angles from degrees to radians
        dssp_feature = np.concatenate(
            [np.sin(radian), np.cos(radian), ASA_SS], axis=1
        )  # Transform angles to sin and cos

        return dssp_feature  # Return transformed DSSP features

    def get_dssp(data_path, dssp_path, ID, ref_seq):
        try:
            os.system(
                "{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID)
            )  # Run DSSP to generate DSSP file

            dssp_seq, dssp_matrix = process_dssp(
                dssp_path + ID + ".dssp"
            )  # Process the DSSP output
            if dssp_seq != ref_seq:  # If sequences do not match
                dssp_matrix = match_dssp(
                    dssp_seq, dssp_matrix, ref_seq
                )  # Align and match DSSP features
            np.save(
                dssp_path + ID + "_dssp.npy", transform_dssp(dssp_matrix)
            )  # Save the transformed DSSP features

            print(
                transform_dssp(dssp_matrix).shape
            )  # Print the shape of the transformed DSSP features
            print(len(ref_seq))  # Print the length of the reference sequence

            os.system(
                "rm {}.dssp".format(dssp_path + ID)
            )  # Remove the DSSP file after processing
            return 0
        except Exception as e:  # Catch exceptions
            print(e)  # Print exception message
            return None

    pdbfasta = {}  # Initialize dictionary to map PDB file names to sequences
    with open(fasta_file) as r1:  # Open the FASTA file
        fasta_ori = r1.readlines()  # Read all lines
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":  # If the line is a header line
            name = (
                fasta_ori[i].split(">")[1].replace("\n", "")
            )  # Extract the name of the sequence
            seq = fasta_ori[i + 1].replace("\n", "")  # Extract the sequence
            pdbfasta[name] = seq  # Map the name to the sequence

    fault_name = (
        []
    )  # Initialize list to store names of sequences that failed DSSP processing
    for name in pdbfasta.keys():  # Iterate through each PDB name
        sign = get_dssp(
            pdb_path, dssp_path, name, pdbfasta[name]
        )  # Run DSSP processing for each PDB
        if sign == None:
            fault_name.append(name)  # Add the PDB name to the fault list
    if fault_name != []:  # If there are any faults
        np.save(
            "../Example/structure_data/dssp_fault.npy", fault_name
        )  # Save the list of faults


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate features from protein sequences."
    )  # Create an argument parser

    # Define command-line arguments
    parser.add_argument(
        "--fasta_file", type=str, default="../data/nrPDB-GO_sequences.fasta"
    )
    parser.add_argument(
        "--prottrans_output_path", type=str, default="../data/afDataSet/prottrans/"
    )
    parser.add_argument(
        "--esm2_output_path", type=str, default="../data/afDataSet/esm2/"
    )
    parser.add_argument("--pdb_dir", type=str, default="../data/af_pdb_dir/")
    parser.add_argument(
        "--proc_pdb_output_path", type=str, default="../data/afDataSet/structure_data/"
    )
    parser.add_argument(
        "--dssp_output_path", type=str, default="../data/afDataSet/dssp/"
    )

    args = parser.parse_args()  # Parse the command-line arguments

    get_prottrans(
        args.fasta_file, args.prottrans_output_path
    )  # Run ProtTrans feature extraction

    proc_pdb(
        args.fasta_file, args.pdb_dir, args.proc_pdb_output_path
    )  # Process PDB files

    get_dssp(
        args.fasta_file, args.pdb_dir, args.dssp_output_path
    )  # Run DSSP processing


if __name__ == "__main__":
    main()  # Call the main function if this script is executed
