# Importance Metrics for Omics with Deep Learning

[DISCLAIMER](./disclaimer.md)

[LICENSE](./license.md)

This repo contains:
- Scripts and notebooks to run various experiments regarding SHAP values in multi-view, multi-omics deep learning models.
- A package with model definitions for two multi-view multi-omics models as well as helpers for investigations into importance metrics.

Includes a pytorch rewrite of:  Lee, C. & van der Schaar, M. A Variational Information Bottleneck Approach to Multi-Omics Data Integration. Preprint at https://doi.org/10.48550/arXiv.2102.03014 (2021).  Original code at https://github.com/chl8856/DeepIMV.  Other models include a non-variational model that averages the latent representations of the different views, and a simple MLP that concatenates the views.

## Requirements
The python packages in requirements.txt should be all you need (plus Jupyter for notebooks).  Install in a virtual environment/conda environment with:

```
pip install -r requirements.txt
```

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

A script to run the pipeline from model training to variable importance is at `lambda/lambda_function.py`.  This can be run from the command line with the requirements installed and `diomics` installed or on your `PYTHONPATH` environment variable.  

```bash
export PYTHONPATH=/path/to/project/folder

python lambda/lambda_function.py --do-train --do_eval --fmeta-sample_names SampleID_proteomics SampleID_lipidpos SampleID_metab --fmeta-target-name Virus --edata-filenames OMICS_ICL104_Metabolomics_YMK.csv ICL104_lipids_aligned_for_stats.csv ICL104_proteins_luke.csv --edata-cnames Metabolite Name Protein --shapley --igrads
```

The equivalent command can be passed as a string to `system` in R, or using `system2`:

```r
system2("python", args = c("lambda/lambda_function.py", "--do-train", "--do-eval", "--fmeta-sample-names", "SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab", "--fmeta-target-name", "Virus", "--edata-filenames", "OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv", "--edata-cnames", "Metabolite", "Name", "Protein", "--shapley", "--igrads"))
```

# Lambda Container

## Build and run
A containerized version of this code is in the `lambda/` folder.  The container can be built by installing docker and calling:

```bash
docker build -t diomics:latest .
```

Once the image is built, you can test it locally by running and pinging a container.  First, we run the container, mounting the data folder to the container's `/var/task/data` folder, and exposing port 9000 mapping to the container's port 8080.

```bash
docker run -p 9000:8080 -v /Users/clab683/git_repos/DeepIMV/data:/var/task/data diomics:latest
```

The mounted data folder must contain the data files that will be used for training and testing.  The folder structure looks like:
```
data/
    fmeta.csv
    edata/
        edata1.csv
        edata2.csv
        ...
    model.pt
```

With `fmeta.csv` being a valid f_meta file containing sample name alignment for the various edata files in the `edata` subfolder.  `model.pt` must only be included if the `--do-train` flag is not specified and must be a model that is amenable to dimension of the data passed in.

## Test using curl/wget
And tested using a curl/wget command similar to the example from the AWS docs, but passing in arguments using the `--data` flag. 

**httr2**:

```r
library(httr2)

url <- "http://localhost:9000/2015-03-31/functions/function/invocations"

req <- request(url)

headers = c(
  'Content-Type' = 'text/plain'
)

body = list(
    do_train = "true",
    do_eval = "true",
    fmeta_sample_names = c("SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"),
    fmeta_target_name = "Virus",
    edata_filenames = c("OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"),
    edata_cnames = c("Metabolite", "Name", "Protein"),
    shapley = "true",
    igrads = "true"
)

req <- req |>
    req_headers(
        'Content-Type' = 'text/plain'
    )  |>
    req_body_json(body)

result <- req |> req_method("GET") |> req_perform()
```

**curl:**

```bash
curl --location --request GET 'http://localhost:9000/2015-03-31/functions/function/invocations' \
--header 'Content-Type: text/plain' \
--data '{
    "do_train": "true",
    "do_eval": "true",
    "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
    "fmeta_target_name": "Virus",
    "edata_filenames": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
    "edata_cnames": ["Metabolite", "Name", "Protein"],
    "shapley": "true",
    "igrads": "true"
}'
```

**wget:**

```bash
wget --no-check-certificate --quiet \
  --method GET \
  --timeout=0 \
  --header 'Content-Type: text/plain' \
  --body-data '{
    "do_train": "true",
    "do_eval": "true",
    "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
    "fmeta_target_name": "Virus",
    "edata_filenames": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
    "edata_cnames": ["Metabolite", "Name", "Protein"],
    "shapley": "true",
    "igrads": "true"
}' \
   'http://localhost:9000/2015-03-31/functions/function/invocations'
```
