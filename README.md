# Importance Metrics for Omics with Deep Learning

[DISCLAIMER](./disclaimer.md)

[LICENSE](./license.md)

This repo contains:
- Scripts and notebooks to run various experiments regarding SHAP values in multi-view, multi-omics deep learning models.
- A package with model definitions for two multi-view multi-omics models as well as helpers for investigations into importance metrics.

Includes a pytorch rewrite of:  Lee, C. & van der Schaar, M. A Variational Information Bottleneck Approach to Multi-Omics Data Integration. Preprint at https://doi.org/10.48550/arXiv.2102.03014 (2021).  Original code at https://github.com/chl8856/DeepIMV.  Other models include a non-variational model that averages the latent representations of the different views, and a simple MLP that concatenates the views.

## Requirements
Dependencies are maintained using [uv](https://docs.astral.sh/uv/) in `pyproject.toml`, `uv.lock` and `requirements.txt`.  

### Using uv (recommended)
Install dependencies using uv, which provides faster package resolution and installation.

```bash
# Install uv if you haven't already

# Linux/Macos, see docs for other options
curl -LsSf https://astral.sh/uv/install.sh | sh

# optionally: pip install uv

# Create a virtual environment and install dependencies
uv sync

# install all extra dependencies, e.g. for running experiments
uv sync --all-extras

# activate the environment
source .venv/bin/activate
```

See [this section](https://docs.astral.sh/uv/pip/environments/) of the `uv` docs for how to work with python environments with uv, including how to install to a different environment (e.g. conda).

### Using pip
Alternatively, install in a virtual environment/conda environment with the below commands.  Note that all of these can be prepended with `uv` to use uv in the installation (.e.g `uv pip install -e .`).

```bash
# from pyproject.toml
pip install -e .

# from pyproject.toml, with optional dependencies for running experiments
pip install -e .[experiments]

# from requirements.txt
pip install -r requirements.txt
```

### Quarto Notebooks
The notebooks are in quarto (.qmd) format.  Install quarto from their website https://quarto.org/docs/get-started/.  VSCode has a quarto plugin that allows you to run code chunks from a .qmd document in an interactive window.  Quarto can also convert .qmd to .ipynb from the command line with e.g. `quarto convert myfile.qmd`.

## Ablation Experiments
We perform experiments to assess the effect of layer size and random noise on SHAP values.  See [the README for the scripts used to run these](bin/README.md).

## Variable importance

We take a look at SHAPley values using the `shap` package, as well as integrated gradients using our own implementation.  These are compared with each other and the results of a simple ANOVA model.

## Data
Most development was done using the ICL-104 dataset.  Please email if you need access to this dataset.  Any other multi-view omics dataset should work, though modifications will need to be made to the path and probably certain details of the model (e.g. if you have only 2 views instead of 3).

## Code
For some examples of how to use the code, see the first couple sections of `notebooks/deep_imv_examples.ipynb` for the DeepIMV rewrite.  Other notebooks include examples on how to train a model and perform variable importance, 

To try to recreate the tcga results from the deepIMV paper:
1.  Get the data.  `bin/get_tcga.py` uses wget to pull from the Broad Institute website.
2. Preprocess the data using my modified version of their notebook:  `notebooks/data_processing_TCGA.ipynb`, which will output their kernel-PCA representation of the four data types.
3.  In the TCGA section of `notebooks/deep_imv_example.ipynb`, you can train and test the model using the data you just created.

**Multi-view MLP**
The multi-view MLP implementation is in `models/multi_mlp.py`.  It is very similar to the DeepIMV architecture, but does not use any variational inference.  Instead it simply computes an average of the latent representations of the different views, and then uses that as input to the the final MLP.  An example of running this model is in `notebooks/multiview_mlp_example.ipynb`.

## Performing Training/Getting Importance Scores From a Script

A script to run the pipeline from model training to variable importance is at `bin/run_pipeline.py`.  This can be run from the command line with the requirements installed and `diomics` installed or on your `PYTHONPATH` environment variable.  

```bash
# if importing from the project directory directly
# export PYTHONPATH=/path/to/project/folder

python bin/run_pipeline.py --do-train --do_eval --fmeta-sample_names SampleID_proteomics SampleID_lipidpos SampleID_metab --fmeta-target-name Virus --edata-filenames OMICS_ICL104_Metabolomics_YMK.csv ICL104_lipids_aligned_for_stats.csv ICL104_proteins_luke.csv --edata-cnames Metabolite Name Protein --shapley --igrads
```

The equivalent command can be passed as a string to `system` in R, or using `system2`:

```r
system2("python", args = c("bin/run_pipeline.py", "--do-train", "--do-eval", "--fmeta-sample-names", "SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab", "--fmeta-target-name", "Virus", "--edata-filenames", "OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv", "--edata-cnames", "Metabolite", "Name", "Protein", "--shapley", "--igrads"))
```

# Lambda Container (DEPRECATED)

This is not being maintained.  I leave documentation in `lambda/README.md` in case there is interest in deploying the pipline in an AWS lambda container.


