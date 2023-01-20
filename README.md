# AMBROSIA
AMBROSIA performs prediction of carbohydrate binding residues from a given protein sequence. It encodes protein sequence using embedding derived from the pre-trained protein language model ESM-2 and convolutional neural network to provide predictions in two classes: 1 - residue participates in carbohydrate binding site formation and 0 - it does not. We consider a residue to form carbohydrate binding site if at least one of its heavy atoms is located closer than 7 A to any atom of the carbohydrate ring.

## Installation
### Prerequisites
- numpy
- torch
- pandas
- h5py
- tqdm
## Procedure
- First, generate residue-level embeddings of your protein chains using pre-trained ESM-2 pLM (see https://github.com/facebookresearch/esm for more informations). Current models have been optimized for pre-trained models `esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D` `esm2_t30_150M_UR50D` and `esm2_t33_650M_UR50D`. 
- Store these embeddings using the hdf5 format with one entry per chain where the key is the chain label and the value the embedding.
- Locate the model you need for the prediction. For instance for model `esm2_t33_650M_UR50D` it is `model/esm2_t33_lr5e-7_bs4096.pth`
- Run your predictions using the command:
``` bash
python predict_ambrosia.py path/to/file/embeddings.h5 model/model.pth results.csv
```
- The output is in csv format with one colon corresponding to the Chain label, one to the index of the residue of interest (0-based) , the prediction and the probability of this prediction.
## Test
For instance to run the prediction on protein chains 3OT9\_A and 2RDK\_A, use the example files:
``` bash
python predict_ambrosia.py examples/test_set.h5 model/esm2_t33_lr5e-7_bs4096.pth results/test_results.csv
```
