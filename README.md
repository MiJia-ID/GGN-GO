# GGN-GO: A Geometric Graph Network Method for Protein Function Prediction Using Multi-scale Structural Features

[![DOI](https://zenodo.org/badge/818847334.svg)](https://zenodo.org/doi/10.5281/zenodo.13768952)
![Figure.1](./Model/fig1.jpg)
## Installation

Clone the current repo

    git clone https://github.com/MiJia-ID/GGN-GO.git
    conda env create -f environment.yml
    pip install . or pip install -r requirements.txt

You also need to install the relative packages to run ESM2 and ProtTrans protein language model. \


## Run GGN-GO for prediction
Simply run:  
```
python predict.py --fasta_file ../Predict/6GEN-G.fasta
```
And the prediction results will be saved in  
```
../Predict/
```
The default model parameters are trained on the combination of PDBch and AFch training set, e.g., `model_bp.pt`, `model_cc.pt` and `model_mf.pt`.\
You can also use the model parameters which are only trained on the PDBch training set, e.g., `model_bp_pdb.pt`, `model_cc_pdb.pt` and `model_mf_pdb.pt`.

### output
```txt
### Predictions made by GGN-GO.			
Protein	GO_term	    Score	GO_term_name
6GEN-G	GO:0003677	0.96044	DNA binding
6GEN-G	GO:0046982	0.95577	protein heterodimerization activity
6GEN-G	GO:0046983	0.93472	protein dimerization activity
```

## Dataset and model
We provide the datasets here for those interested in reproducing our paper. The datasets used in this study are stored in ```../Data/```.
The trained GGN_GO models can be found under ```../Model/```.

## Model training
Using the pdb database as an example, the detailed steps of downloading the dataset, generating features and training for this project are presented

### 1.Download the PDB dataset based on the fasta file
Download the corresponding PDB files from [Protein Data Bank](http://www.rcsb.org/pdb/files/) based on `nrPDB-GO_sequences.fasta` and save them to the ```../data/pdb_dir/``` path.

### 2.Process the PDB files to generate feature files
Use the following two scripts to generate the feature files required for training:
```
python my_Features.py
python ESM2.py
```
1. dssp features ```../data/bpDataSet/dssp/```
2. ESM2 features ```../data/bpDataSet/esm2/```
3. ProtTrans features ```../data/bpDataSet/prottrans/```
4. structure features ```../data/bpDataSet/structure_data/```

This step requires downloading ESM2 and ProtTrans.
#### ProtTrans
You need to prepare the pretrained language model ProtTrans to run GLMSite:  
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)).  
#### ESMFold
The protein structures should be predicted by ESMFold to run GLMSite:  
Download the ESMFold model ([guide](https://github.com/facebookresearch/esm))


### 3.Model training
Use train.py for training:
 ```
python train.py --device 0
                --task bp 
                --batch_size 64 
                --contrast True
 ```
The generated parameter files will be saved to  ```../Model/ ```.




