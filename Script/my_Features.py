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
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1')
    args = parser.parse_args()
    gpu = args.gpu

    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()  
        if line and line[0] == ">":  
            ID_list.append(line[1:]) 
        elif line: 
            seq_list.append(" ".join(list(line))) 

    for id, seq in zip(ID_list[:9], seq_list[:9]):
        print(f"ID: {id}")
        print(f"Sequence: {seq[:]}...")
        print("len:",len(seq))

    model_path = "./Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval().to(device)

    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]

        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()


        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]

            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)


def get_pdb_xyz(pdb_file, ref_seq):
    current_pos = -1000
    X = []
    current_aa = {}  # 'N', 'CA', 'C', 'O'
    try:
        for line in pdb_file:
            if (line[0:4].strip() == "ATOM" and (line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = (line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom in ['N', 'CA', 'C', 'O']:
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
    except:
        print("111")
        return None
    if len(X) == len(ref_seq):
        return np.array(X)
    else:
        print("len(X)",len(X))
        print("len(ref_seq)", len(ref_seq))
        return None


def proc_pdb(fasta_file, pdb_path, output_path):
    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n', '')
            seq =fasta_ori[i + 1].replace('/n', '')
            pdbfasta[name] = seq.replace('\n','')

    for key in pdbfasta.keys():
        with open(pdb_path + key + '.pdb', 'r') as r1:
            pdb_file = r1.readlines()
        coord = get_pdb_xyz(pdb_file, pdbfasta[key])
        if coord is None:
            print("pdb:",key)
        np.save(output_path + key + '.npy', coord)


def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = './dssp'

    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9) 
            SS_vec[SS_type.find(SS)] = 1
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9)  
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:, 0:2]
        ASA_SS = dssp_feature[:, 2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)

        return dssp_feature

    def get_dssp(data_path, dssp_path, ID, ref_seq):
        try:
            os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID))

            dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            if dssp_seq != ref_seq:
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
            np.save(dssp_path + ID + "_dssp.npy", transform_dssp(dssp_matrix))

            print(transform_dssp(dssp_matrix).shape)
            print(len(ref_seq))

            os.system('rm {}.dssp'.format(dssp_path + ID))
            return 0
        except Exception as e:
            print(e)
            return None

    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n', '')
            seq = fasta_ori[i + 1].replace('\n', '')
            pdbfasta[name] = seq

    fault_name = []
    for name in pdbfasta.keys():
        sign = get_dssp(pdb_path, dssp_path, name, pdbfasta[name])
        if sign == None:
            fault_name.append(name)
    if fault_name != []:
        np.save('../Example/structure_data/dssp_fault.npy', fault_name)



import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate features from protein sequences.")

    parser.add_argument("--fasta_file", type=str, default='../data/nrPDB-GO_sequences.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='../data/afDataSet/prottrans/')
    parser.add_argument("--esm2_output_path", type=str, default='../data/afDataSet/esm2/')
    parser.add_argument('--pdb_dir', type=str, default='../data/af_pdb_dir/')
    parser.add_argument("--proc_pdb_output_path", type=str, default='../data/afDataSet/structure_data/')
    parser.add_argument("--dssp_output_path", type=str, default='../data/afDataSet/dssp/')

    args = parser.parse_args()

    get_prottrans(args.fasta_file, args.prottrans_output_path)

    proc_pdb(args.fasta_file,args.pdb_dir,args.proc_pdb_output_path)

    get_dssp(args.fasta_file, args.pdb_dir, args.dssp_output_path)


if __name__ == "__main__":
    main()