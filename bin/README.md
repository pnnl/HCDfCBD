# SHAP Value Architecture Experiments

We run experiments regarding the effects of layer sizes and random noise on SHAP values, specifically how they affect rankings of biomolecules for the purposes of gleaning insight into the biological system.  To run any of the examples below, make sure you've installed everything in `requirements.txt`.

## Data
The data used in this experiment is provided as expression data along with a target column of virus type, split into train and test sets.  Specifically, the following files *for both ICL102 and ICL104*:

- `lip_neg_train.csv`, `lip_neg_test.csv` 
- `metab_train.csv`, `metab_test.csv` 
- `pro_train.csv`, `pro_test.csv`

Put these files in a folder of your choice.  The path of this folder will be specified when running scripts and it will be expected that these files (with the same names) will be found there.

## Running Scripts

The experiments are organized via .yaml config files under `bin/cfg`.  These define parameters which are ingested into the experiment scripts via [hydra](https://hydra.cc/docs/intro/), essentially a fancy argparse.  The two immediate files in `bin/cfg` are the base configurations which specify default values.  For example, `bin/noise_experiments.py` is the entrypoint which ingests the config `bin/cfg/config-noise.yaml`.  Arguments can be changed similarly to running a python script, make sure to point the scripts to the folder you put the data in:

```bash
# in base project directory
export PYTHONPATH=.

# use default values in bin/cfg/config-noise.yaml
python bin/noise_experiments.py data.base_path=<your-data-folder-path>

# change two parameters specified in the base config
python bin/noise_experiments.py n_evals_per_noise=15 n_iters=1500 data.base_path=<your-data-folder-path>
```

To perform the full experiments, the configs in `bin/cfg/hydra/sweeper` define sweeps over various parameters such as layer sizes, combination schemes, and added noise levels in our experiments.  For example, the config `bin/cfg/hydra/sweeper/noise-sweep-binary-3view.yaml` is used to do the experiments where we vary layer size and noise in parallel with all 3 datasets to see the effect:

```yaml
# @package _global_
# @package hydra.sweeper
_target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
max_batch_size: null
params:
  layer_x_noise: "{layer_sizes: [[128, 128], [256, 256], [512, 512]], n_noise:100}, {layer_sizes:[[128, 128], [256, 256], [512, 512]], n_noise:500}, {layer_sizes:[[128, 128], [256, 256], [512, 512]], n_noise:1000}, {layer_sizes:[[128, 128], [256, 256], [512, 512]], n_noise:2500}, {layer_sizes:[[128, 128], [256, 256], [512, 512]], n_noise:4000}, {layer_sizes:[[148, 148], [219, 219], [438, 438]], n_noise:100}, {layer_sizes:[[192, 192], [169, 169], [338, 338]], n_noise:500}, {layer_sizes:[[223, 223], [145, 145], [290, 290]], n_noise:1000}, {layer_sizes:[[276, 276], [117, 117], [234, 234]], n_noise:2500}, {layer_sizes:[[309, 309], [105, 105], [210, 210]], n_noise:4000}"
  datasets: ['metab', 'lip', 'pro']
  experiment_name: noise-exp-3view-binary
```
This config sweeps over many values of the `layer_x_noise` argument, which specifies levels of noise and layer sizes for each training run.  It will inherit all unspecified config options from the base config.  To run these sweeps:

```bash
# again, adding `data.base_path=<your-data-folder-path>` to each command

# use the above sweep config
python bin/noise_experiments.py -m hydra/sweeper=noise-sweep-binary-3view

# another config which specifies only metabolites and lipids
python bin/noise_experiments.py -m hydra/sweeper=noise-sweep-metab-v-lip

# make it run faster with ray
python bin/noise_experiments.py -m hydra/sweeper=noise-sweep-metab-v-lip hydra/launcher=ray

# override the experiment name to keep these runs separate
python bin/noise_experiments.py -m hydra/sweeper=noise-sweep-metab-v-lip experiment_name=my_experiment_1
```

The experiment which uses the other base config `bin/cfg/config.yaml` uses `bin/train.py` as an entrypoint.  Sweep configs that use this entrypoint will have arguments distinct to that base config.

## MlFlow

Each script logs results with [mlflow](https://mlflow.org/docs/latest/getting-started/index.html).  For our purposes mlflow is simply a way to automatically organize the results and output of experiments in folders which can be viewed in a ui and retrieved with a python api.

By default, mlflow will log results to a folder called `mlruns` in the directory you were in when you launched the script (in our case probably the project directory root).  The contents of this folder are not intuitive to navigate for a human, as the actual contents of each folder is not given in the name.  The easiest way to view the contents is to start the mlflow UI from the directory where `mlruns` sits:

```bash
mlflow ui
```

By default this will start the mlflow UI running at `localhost:5000`.  Navigate to that address in a browser and you will see something resembling the following UI:

![](../www/mlflow-ex.png)

On the left are experiments, and in the main panel the results of various runs, with metrics that we logged.

Clicking on a run will allow you to browse its results and artifacts (outputs like model weights, matrices of SHAP values etc.).  Examples of retrieving and inspecting artifacts/metrics from various experiments/runs are shown in the notebooks:

- `notebooks/check_by_feature_variability.qmd`
- `notebooks/inspect_mlflow_runs.qmd`

## Descriptions of Experiments Defined by Sweeper Configs

We had two main experiments, one that measured performance across architectures with different layer sizes, and either mean or concatentation

- Performance differences between concat/mean combination and layer sizes
    - `concat-binary.yaml`:  Different layer sizes with a concatenation scheme on the binary ICL104 dataset
    - `concat-multiclass.yaml`:  Different layer sizes with a concatenation combination scheme on the multiclass ICL102 dataset 
    - `mean-binary.yaml`:  Different layer sizes with a mean combination scheme on the binary ICL104 dataset.
    - `mean-multiclass.yaml`:  Different layer sizes with a mean combination scheme on the binary ICL104 dataset

- Variation in SHAP values due to added noise and layer size differences, with 10 runs per sweep parameter. ICL 104 only.
    - `noise-sweep-binary-3view.yaml`:  Collects SHAP values using all 3 views (metabolites, lipids, proteomics), with metabolites receiving all augmented noise.
    - `noise-sweep-binary-metab-v-lip/noise-sweep-binary-metab-v-lip-2` SHAP values for a network with metabolites and lipids using two different layer size adjustment schemes
    - `noise-sweep.yaml`:  Base experiment with metabolites and proteins
    - `noise-sweep-small.yaml`:  Similar to base experiment, metabolites and proteins with smaller initial layer size for the protein network. 
