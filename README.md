GGN-GO
====
GGN-GO: A Geometric Graph Network Method for Protein Function Prediction Using Multi-scale Structural Features
---

<img src="Model/fig1.jpg">

## Installation

Clone the current repo

    git clone https://github.com/MiJia-ID/GGN-GO.git
    conda env create -f environment.yml
    pip install . or pip install -r requirements.txt

You also need to install the relative packages to run ESM2 and ProtTrans protein language model. \
ESM2 model weight we use can be downloaded [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt).
ProtTrans model weight we use can be downloaded [here](https://github.com/agemagician/ProtTrans)

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



