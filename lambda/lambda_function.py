import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from diomics.models.multi_mlp import simple_FC, JointMLP
from diomics.featimportance.igrads import integrated_grads
from diomics.featimportance.comparison import overlap_sets
from diomics.featimportance.plot_helpers import igrad_beeswarm_plot, make_igrad_plot_df
from diomics.featimportance.shapley import JointMLPWrapper

import shap
import numpy as np
import pandas as pd
import copy
import datetime
import logging
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Dummy event object.  Passed as the 'data' field in curl

# event['fmeta_sample_names'] = ['SampleID_proteomics', 'SampleID_lipidpos', 'SampleID_metab']
# event['fmeta_target_name'] = "Virus"
# event['edata_cnames'] = ['Metabolite', 'Name', "Protein"]

# event = {
#     "do_train": "true",
#     "do_eval": "true",
#     "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
#     "fmeta_target_name": "Virus",
#     "edata_filenames": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
#     "edata_cnames": ["Metabolite", "Name", "Protein"],
#     "shapley": "true",
#     "igrads": "true"
# }

def handler(event, context):
    logging.info("event: %s", event)

    # flags to do training and/or evaluation
    do_train = event.get('do_train', False)
    do_eval = event.get('do_eval', False)
    
    # get absolute paths of files in /data/edata
    edata_paths = [os.path.join("data/edata", path) for path in event['edata_filenames']]
    fmeta_path = os.path.join("data/fmeta.csv")    

    # load data
    datas = []

    for fname,path in zip(event['edata_filenames'], edata_paths):
        if os.path.splitext(path)[1] == '.csv':
            datas.append(pd.read_csv(path))
        elif os.path.splitext(path)[1] == '.xlsx':
            datas.append(pd.read_excel(path))
        else:
            raise ValueError("Unknown file type: {}, must be one of .csv or .xlsx".format(path))

        if any(datas[-1].columns.isin(['KEGG', 'CAS', 'PubChem'])):
            datas[-1] = datas[-1].drop(columns=['KEGG', 'CAS', 'PubChem'])

    fmeta = pd.read_csv(fmeta_path)
    fmeta = fmeta[~fmeta[event['fmeta_sample_names']].isna().any(axis=1)]

    # Placeholder for example DHS data TODO:  probably remove this unless we want them to be able to pass arguments to .replace()
    fmeta[event['fmeta_target_name']] = fmeta[event['fmeta_target_name']].replace({'Cal[0-9]*': 'Cal'}, regex=True)

    for edata_cname, k in zip(event['edata_cnames'], range(len(datas))):
        datas[k] = datas[k].set_index(edata_cname)
        datas[k][datas[k] == 0] = np.nan
        if not any(datas[k] < 0):
            datas[k] = np.log2(datas[k])

        datas[k] = datas[k] - datas[k].median(axis=0)

        # 0-1 normalize the data
        datas[k] = (datas[k] - np.min(datas[k])) / (np.max(datas[k]) - np.min(datas[k]))
        
        tmp_inds = (datas[k].isnull() == False).sum(axis = 1) > 0
        datas[k] = datas[k][tmp_inds]
        datas[k] = datas[k].apply(lambda row: row.fillna(row.mean()), axis=1)

    # TODO:  Make this flexible, currently hard-coded for ICL-104 dataset
    y = pd.Series(['Cal' if 'Cal04' in el else 'Mock' for el in datas[0].columns])

    np.random.seed(1565)
    cal_inds = np.random.choice(np.where(y == 'Cal')[0], int(np.sum(y == 'Cal') * 0.2))
    mock_inds = np.random.choice(np.where(y == 'Mock')[0], int(np.sum(y == 'Mock') * 0.2))
    test_inds = np.concatenate([cal_inds, mock_inds])

    train_inds = np.isin(np.arange(y.shape[0]), test_inds) == False

    train_splits = {}
    test_splits = {}

    for k in range(len(datas)):
        train_splits[k] = datas[k].iloc[:,train_inds]
        test_splits[k] = datas[k].iloc[:,test_inds]

    ytrain = y.iloc[train_inds]
    ytest = y.iloc[test_inds]

    last_hidden_size = 64

    marginal_models = []
    for k in range(len(datas)):
        input_size = datas[k].shape[0]
        marginal_models.append(simple_FC(input_size=input_size, hidden_sizes=[128, last_hidden_size], prediction_dim=y.nunique()))

    # joint model
    joint_model = JointMLP(marginal_models=marginal_models, hidden_dim=128)

    # define optimizer
    optimizer = AdamW(joint_model.parameters(), lr=1e-4)

    tensors = {}
    for k in train_splits.keys():
        tensors[k] = torch.tensor(train_splits[k].values, dtype=torch.float32).T

    y_gt = ytrain.astype('category').cat.codes
    y_gt = torch.tensor(y_gt.values, dtype=torch.int64)

    views = list(tensors.values())
    
    if do_train:
        for i in range(1500):
            views = list(tensors.values())

            if np.random.rand() < 0.1:
                idx = np.random.randint(0, len(views))
                views = [v if i != idx else None for i, v in enumerate(views)]

            yhat, h, yhats, hiddens = joint_model(views)

            loss = joint_model.loss(y_gt, yhat, yhats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 2.0)
            optimizer.step()

            print(f'Iteration {i+1} loss: {loss.item():.3f}')

        # save the pytorch model
        torch.save(joint_model.state_dict(), 'data/model.pt')
    else:
        # load the pytorch model
        joint_model.load_state_dict(torch.load('data/model.pt'))

    joint_model.eval()

    if do_eval:
        ytest_gt = ytest.astype('category').cat.codes

        test_tensors = {}
        for k in test_splits.keys():
            test_tensors[k] = torch.tensor(test_splits[k].values, dtype=torch.float32).T

        with torch.inference_mode():
            yhat, poe_dist, yhats, dists = joint_model(list(test_tensors.values()))

        ypred = yhat.argmax(dim=1).numpy()

        ACC_TEST = (ypred == ytest_gt.to_numpy()).mean()

        CONFUSION_TEST = confusion_matrix(ytest_gt, ypred)

        # dump eval results, just one line with the accuracy for now
        with open('data/eval_results.txt', 'w') as f:
            f.write(f"Model test accuracy: {ACC_TEST}\n")

        # dump the confusion matrix
        pd.DataFrame(CONFUSION_TEST).to_csv('data/confusion_matrix.csv', index=False)
        
    # SHAPley values
    if event.get('shapley', False):
        joint_model_wrp = JointMLPWrapper(joint_model)

        # make the explainer object
        explainer = shap.DeepExplainer(joint_model_wrp, views)

        # make tensors of the whole dataset
        all_data_tensors = [torch.tensor(d.values, dtype=torch.float32).T for d in datas]

        # compute the shapley values
        shap_values = explainer.shap_values(all_data_tensors, check_additivity=False)

        # make all the plots and dump them to disk
        # TODO:  I am assuming binary classification, for multi-class we need to loop over the first index as well probably
        for tmp_tensor, sv, fname in zip(all_data_tensors, shap_values[0], event['edata_filenames']):
            fig = plt.gcf()
            shap.summary_plot(sv, features=tmp_tensor, feature_names=datas[k].index, show=False)
            fig.savefig('data/shap_summary_plot_{}.png'.format(fname))
            plt.close(fig)

        abs_shap_values = [np.abs(el).sum(axis = 0) for el in shap_values[0]]
        sorted_shap_values = [sorted(el, reverse=True) for el in abs_shap_values]

        out_dataframes_shap = []

        for i, fname in enumerate(event['edata_filenames']):
            tmp_df = pd.DataFrame({
                'feature': datas[i].iloc[(-abs_shap_values[i]).argsort(), :].index, 
                'shapley_value': sorted_shap_values[i]
            })

            scores_df = pd.DataFrame(shap_values[0][i].T, columns=datas[i].columns)
            scores_df.index = datas[i].index

            tmp_df = tmp_df.merge(scores_df, how = 'left', left_on='feature', right_index=True)
            out_dataframes_shap.append(tmp_df)
            tmp_df.to_csv('data/shapley_values_{}.csv'.format(fname), index=False)

    # Integrated gradients
    # TODO:  Again, here I'm assuming binary classification, we'd need to examine each output dimension for multi-class.
    if event.get('igrads', False):
        baselines = [-torch.rand_like(el)/2 for el in all_data_tensors]
        baselines = [el[:, torch.randperm(el.shape[1])] for el in baselines]

        all_igrad_scores = integrated_grads(all_data_tensors, baselines, joint_model, n_steps=100, class_idx=0)

        abs_igrad_scores = [el.abs().sum(axis=0).numpy() for el in all_igrad_scores]
        sorted_igrad_scores = [sorted(el, reverse=True) for el in abs_igrad_scores]

        out_dataframes_igrad = []

        for i, fname in enumerate(event['edata_filenames']):
            tmp_df = pd.DataFrame({
                'feature': datas[i].iloc[(-abs_igrad_scores[i]).argsort(), :].index, 
                'IGrad_score': sorted_igrad_scores[i]
            })

            scores_df = pd.DataFrame(all_igrad_scores[i].T, columns=datas[i].columns)
            scores_df.index = datas[i].index

            tmp_df = tmp_df.merge(scores_df, how = 'left', left_on='feature', right_index=True)
            out_dataframes_igrad.append(tmp_df)
            tmp_df.to_csv('data/integrated_gradients_{}.csv'.format(fname), index=False)

        # make the plots and dump them to disk
        for temp_df, tmp_data, edata_cname in zip(out_dataframes_igrad, datas, event['edata_cnames']):
            plot_df = make_igrad_plot_df(
                temp_df, tmp_data.reset_index(), num_biomols = 10, 
                id_vars = edata_cname, var_name = "SampleID", 
                value_name = "Input_value"
            )
            
            fig = plt.gcf()
            igrad_beeswarm_plot(
                plot_df, 
                ycol = edata_cname, 
                xcol = "IGrad_value", 
                raw_val_name = "Input_value", 
                palette = "Spectral", 
                xlabel = "Integrated Gradient Scores", 
                ylabel = edata_cname, 
                title = "", 
                size = 4,
                use_stripplot = True
            )
            plt.tight_layout()
            fig.savefig('data/integrated_gradients_{}.png'.format(edata_cname), bbox_inches='tight')
            plt.close(fig)

    return("success")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', action='store_true', help="whether to train the model from scratch")
    parser.add_argument('--do-eval', action='store_true', help="whether to evaluate the model")
    parser.add_argument('--fmeta-sample-names', nargs='*', default=["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"], help="names of columns in fmeta.csv that correspond to the sample IDs in each of the edata files")
    parser.add_argument('--fmeta-target-name', default="Virus", help="name of the column in fmeta.csv that contains the target variable")
    parser.add_argument('--edata-filenames', nargs='*', default=["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"], help="filenames of the edata files")
    parser.add_argument('--edata-cnames', nargs='*', default=["Metabolite", "Name", "Protein"], help="names of the columns in the edata files that contain the biomolecule names")
    parser.add_argument('--shapley', action='store_true', help="whether to compute SHAPley values and plot the result")
    parser.add_argument('--igrads', action='store_true', help="whether to compute integrated gradients and plot the result")
    args = parser.parse_args()

    event = {
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "fmeta_sample_names": args.fmeta_sample_names,
        "fmeta_target_name": args.fmeta_target_name,
        "edata_filenames": args.edata_filenames,
        "edata_cnames": args.edata_cnames,
        "shapley": args.shapley,
        "igrads": args.igrads
    }

    handler(event, None)

if __name__ == "__main__":
    main()
    