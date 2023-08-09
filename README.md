# DeepIMV-pytorch

A pytorch rewrite of:  Lee, C. & van der Schaar, M. A Variational Information Bottleneck Approach to Multi-Omics Data Integration. Preprint at https://doi.org/10.48550/arXiv.2102.03014 (2021).  Original code at https://github.com/chl8856/DeepIMV.

### Requirements
The python packages in requirements.txt should be all you need (plus Jupyter for notebooks).

### Code
For some examples of how to use the code, see the first couple sections of `deep_imv_examples.ipynb`.

To try to recreate their tcga results:
1.  Get the data.  `get_tcga.py` uses wget to pull from the Broad Institute website.
2. Preprocess the data using my modified version of their notebook:  `data_processing_TCGA.ipynb`, which will output their kernel-PCA representation of the four data types.
3.  In the TCGA section of `deep_imv_examples.ipynb`, you can train and test the model using the data you just created.
