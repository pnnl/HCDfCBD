# Importance Metrics for Omics with Deep Learning

A package with various methods for identifying important variables in multi-omics with deep learning.

Includes a pytorch rewrite of:  Lee, C. & van der Schaar, M. A Variational Information Bottleneck Approach to Multi-Omics Data Integration. Preprint at https://doi.org/10.48550/arXiv.2102.03014 (2021).  Original code at https://github.com/chl8856/DeepIMV.  Other models include a non-variational model that averages the latent representations of the different views, and a simple MLP that concatenates the views.

### Requirements
The python packages in requirements.txt should be all you need (plus Jupyter for notebooks).

### Variable importance

We take a look at SHAPley values using the shap package, as well as integrated gradients using our own implementation.  These are compared with each other and the results of a simple ANOVA model.

### Data
Most development was done using the ICL-104 dataset.  Please email if you need access to this dataset.  Any other multi-view omics dataset should work, though modifications will need to be made to the path and probably certain details of the model (e.g. if you have only 2 views instead of 3).

### Code
For some examples of how to use the code, see the first couple sections of `notebooks/deep_imv_examples.ipynb` for the DeepIMV rewrite.  Other notebooks include examples on how to train a model and perform variable importance, 

To try to recreate the tcga results from the deepIMV paper:
1.  Get the data.  `bin/get_tcga.py` uses wget to pull from the Broad Institute website.
2. Preprocess the data using my modified version of their notebook:  `notebooks/data_processing_TCGA.ipynb`, which will output their kernel-PCA representation of the four data types.
3.  In the TCGA section of `notebooks/deep_imv_example.ipynb`, you can train and test the model using the data you just created.

**Multi-view MLP**
The multi-view MLP implementation is in `models/multi_mlp.py`.  It is very similar to the DeepIMV architecture, but does not use any variational inference.  Instead it simply computes an average of the latent representations of the different views, and then uses that as input to the the final MLP.  An example of running this model is in `notebooks/multiview_mlp_example.ipynb`.
