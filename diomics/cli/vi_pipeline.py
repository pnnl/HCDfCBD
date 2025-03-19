import torch
from torch.optim import AdamW

from ..models.multi_mlp import simple_FC, JointMLP
from ..featimportance.igrads import integrated_grads
from ..featimportance.plot_helpers import igrad_beeswarm_plot, make_igrad_plot_df
from ..featimportance.shapley import JointMLPWrapper

import shap
import numpy as np
import pandas as pd
import logging
import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm

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

def train_joint_model(
    joint_model, tensors, y_gt, optimizer, event, do_validation = True, 
    tensors_valid = None, y_gt_valid = None, max_iter = 5000, drop_view_prob = 0):

    best_valid_loss = torch.tensor(torch.inf)
    iter_no_improvement = 0

    valid_loss_list = []
    train_loss_list = []
    pbar = tqdm(range(max_iter))
    valid_loss = torch.tensor(torch.inf)
    best_iter = 0
    reduce_lr = True

    for i in pbar:
        views = list(tensors.values())

        if np.random.rand() < drop_view_prob:
            idx = np.random.randint(0, len(views))
            views = [v if i != idx else None for i, v in enumerate(views)]

        yhat, h, yhats, hiddens = joint_model(*views)

        loss, _, joint_loss = joint_model.loss(y_gt, yhat, yhats)

        optimizer.zero_grad()
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 2.0)
        optimizer.step()

        # validation step
        if do_validation and i % event['validation_frequency'] == 0 and i > 0:
            assert tensors_valid is not None and y_gt_valid is not None, "Validation tensors and ground truth must be provided if do_validation is True"
            joint_model.eval()

            views_valid = list(tensors_valid.values())
            yhat_valid, h_valid, yhats_valid, hiddens_valid = joint_model(*views_valid)
            valid_loss, _, _ = joint_model.loss(y_gt_valid, yhat_valid, yhats_valid)
            valid_loss_list.append(valid_loss.item())

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_iter = i
                print(f'Validation loss improved to {valid_loss.item()}, saving model')
                torch.save(joint_model.state_dict(), event['best_model_path'])
                iter_no_improvement = 0
            else:
                iter_no_improvement += 1

            if iter_no_improvement > event['patience']:
                if reduce_lr:
                    print(f'No improvement in validation loss for {int(event["patience"] * event["validation_frequency"])} iterations, reducing learning rate')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.25
                    reduce_lr = False
                    iter_no_improvement = 0
                else:
                    print(f'No improvement in validation loss for {int(event["patience"] * event["validation_frequency"])} iterations, stopping training')
                    break

            joint_model.train()

        # set the tqdm message:
        pbar.set_description(f'Iteration {i}, training loss: {loss.item():.3f} validation loss: {valid_loss.item():.3f} best validation loss: {best_valid_loss.item():.3f}')

        train_loss_list.append(loss.item())

    out = {
        "joint_model": joint_model,
        "valid_loss_list": valid_loss_list,
        "train_loss_list": train_loss_list,
        "n_iter": i,
        "best_iter": best_iter + 1
    }

    return out

def make_joint_model(event, datas, y):
    last_hidden_size = 64
    marginal_models = []
    for k in range(len(datas)):
        input_size = datas[k].shape[0]
        marginal_models.append(simple_FC(input_size=input_size, hidden_sizes=[128, last_hidden_size], prediction_dim=y.nunique()))

    # joint model
    joint_model = JointMLP(marginal_models=marginal_models, hidden_dim=128)

    return joint_model

def plot_loss_curve(train_loss_list, valid_loss_list, out, event, ax):
    ax.plot(train_loss_list, label='Training loss')
    ax.plot([(i + 1)*event['validation_frequency'] for i in range(len(valid_loss_list))], valid_loss_list, label='Validation loss')
    # draw a vertical line at the best iteration
    ax.axvline(out['best_iter'], color='r', linestyle='--', label=f'Best validation loss {np.min(valid_loss_list):.3f} at iteration {out["best_iter"]}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()

# use google docstring
def subset_fmeta(fmeta, datas, event, inplace = True) -> pd.DataFrame:
    """Subset the fmeta dataframe to only include samples that have observations in all omics types.

    Args:
        fmeta (pd.DataFrame): The metadata dataframe.
        datas (list): A list of pandas dataframes  in p x n format containing the omics data.
        event (dict): A dictionary containing the event data.

    Returns:
        tuple:  A tuple containing 1.  The subsetted metadata dataframe with only samples (rows) that have a matching sample id in all of the expression data files; and 2. A boolean mask indicating which samples have observations in all omics types.
    """
    has_all = pd.Series([True]*fmeta.shape[0])

    for k in range(len(datas)):
        has_all = has_all & fmeta[event['fmeta_sample_names'][k]].isin(datas[k].columns)
        if sum(has_all) == 0:
            raise ValueError("Found zero samples with observations across all omics types.  Make sure your data has at one complete multi-omics sample and that the sample id columns of fmeta contain entries that match the columns of your expression data files.")

    if inplace:
        fmeta = fmeta[has_all]

    return fmeta, has_all

def naive_preprocess(datas, fmeta, event):
    for k in range(len(datas)):
        datas[k] = datas[k].set_index(event['edata_cnames'][k])
        datas[k][datas[k] == 0] = np.nan
        if not any(datas[k] < 0):
            datas[k] = np.log2(datas[k])

        datas[k] = datas[k] - datas[k].median(axis=0)

        # 0-1 normalize the data
        datas[k] = (datas[k] - np.min(datas[k])) / (np.max(datas[k]) - np.min(datas[k]))
        
        tmp_inds = (datas[k].isnull() == False).sum(axis = 1) > 0
        datas[k] = datas[k][tmp_inds]
        datas[k] = datas[k].apply(lambda row: row.fillna(row.mean()), axis=1)

    y = fmeta[event['fmeta_target_name']]

    # raises error here if your fmeta sample id columns and emeta columns are completely wack.
    fmeta, has_all = subset_fmeta(fmeta, datas, event)

    for k in range(len(datas)):
        datas[k] = datas[k][fmeta[event['fmeta_sample_names'][k]]]

    y = y[has_all]

    return datas, y

def run_pipeline(event, context):
    """A function to run training and variable importance for the multi-mlp model, with arguments formatted for an AWS lambda.
    """
    logging.info("event: %s", event)

    os.makedirs(event['output_dir'], exist_ok=True)

    out_dict = {}

    # flags to do training and/or evaluation
    do_train = event.get('do_train', False)
    do_eval = event.get('do_eval', False)
    
    # get absolute paths of files in /data/edata
    edata_paths = [os.path.join(event['data_dir'], path) for path in event['edata_filenames']]
    fmeta_path = os.path.join(event['data_dir'], event['fmeta_file_name'])    

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

    if event['preprocess']:
        fmeta, y = naive_preprocess(datas, fmeta, event)
    else:
        # just checking that fmeta has at least one valid row
        _ = subset_fmeta(fmeta, datas, event, inplace = False)
        y = fmeta[event['fmeta_target_name']]

    # get train-test split stratified by class
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=1565)
    valid_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1555)

    train_inds, test_inds = next(splitter.split(np.zeros(y.shape), y))
    test_inds_new, valid_inds = next(valid_splitter.split(np.zeros(y.iloc[test_inds].shape), y.iloc[test_inds]))

    valid_inds = test_inds[valid_inds]
    test_inds = test_inds[test_inds_new]

    train_splits = {}
    valid_splits = {}
    test_splits = {}

    for k in range(len(datas)):
        train_splits[k] = datas[k].iloc[:,train_inds]
        valid_splits[k] = datas[k].iloc[:,valid_inds]
        test_splits[k] = datas[k].iloc[:,test_inds]

    out_dict['train_splits'] = train_splits
    out_dict['test_splits'] = test_splits

    ytrain = y.iloc[train_inds]
    yvalid = y.iloc[valid_inds]
    ytest = y.iloc[test_inds]

    # joint model
    joint_model = make_joint_model(event, datas, y)

    # define optimizer
    optimizer = AdamW(joint_model.parameters(), lr=5e-5)

    tensors = {}
    tensors_valid = {}

    for k in train_splits.keys():
        tensors[k] = torch.tensor(train_splits[k].values, dtype=torch.float32).T
        tensors_valid[k] = torch.tensor(valid_splits[k].values, dtype=torch.float32).T

    out_dict['tensors'] = list(tensors.values())

    y_gt = ytrain.astype('category').cat.codes
    y_gt = torch.tensor(y_gt.values, dtype=torch.int64)
    y_gt_valid = yvalid.astype('category').cat.codes
    y_gt_valid = torch.tensor(y_gt_valid.values, dtype=torch.int64)

    if do_train:
        out = train_joint_model(
            joint_model, tensors, y_gt, optimizer, event, do_validation = True,
            tensors_valid = tensors_valid, y_gt_valid = y_gt_valid, max_iter = event['max_iter']
        )

        valid_loss_list = out['valid_loss_list']
        train_loss_list = out['train_loss_list']

        # save loss curve
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_loss_curve(train_loss_list, valid_loss_list, out, event, ax)
        fig.savefig(os.path.join(event['output_dir'], 'loss_curve.png'))

        # train on the training + validation data
        joint_model = make_joint_model(event, datas, y)

        optimizer = AdamW(joint_model.parameters(), lr=5e-5)

        tensors = {}

        for k in train_splits.keys():
            tensor_train = pd.concat(
                [train_splits[k], valid_splits[k]], axis=1
            )

            tensors[k] = torch.tensor(tensor_train.values, dtype=torch.float32).T

        y_gt = pd.concat([ytrain, yvalid]).astype('category').cat.codes
        y_gt = torch.tensor(y_gt.values, dtype=torch.int64)

        out = train_joint_model(
            joint_model, tensors, y_gt, optimizer, event,
            max_iter = out['best_iter'] + event['validation_frequency'], do_validation = False
        )

        joint_model = out['joint_model']

        # save the pytorch model
        torch.save(joint_model.state_dict(), event['final_model_path'])
    else:
        # load the pytorch model
        joint_model.load_state_dict(torch.load(event['best_model_path']))
    
    out_dict['final_model'] = joint_model
    out_dict['final_model_state_dict'] = joint_model.state_dict()

    joint_model.eval()

    if do_eval:
        ytest_gt = ytest.astype('category').cat.codes

        test_tensors = {}
        for k in test_splits.keys():
            test_tensors[k] = torch.tensor(test_splits[k].values, dtype=torch.float32).T

        with torch.inference_mode():
            yhat, poe_dist, yhats, dists = joint_model(*list(test_tensors.values()))

        ypred = yhat.argmax(dim=1).numpy()

        ACC_TEST = (ypred == ytest_gt.to_numpy()).mean()

        CONFUSION_TEST = confusion_matrix(ytest_gt, ypred)

        # dump eval results, just one line with the accuracy for now
        with open(os.path.join(event['output_dir'], 'eval_results.txt'), 'w') as f:
            f.write(f"Model test accuracy: {ACC_TEST}\n")

        # dump the confusion matrix
        pd.DataFrame(CONFUSION_TEST).to_csv(os.path.join(event['output_dir'], 'confusion_matrix.csv'), index=False)
    
        ACC_TRAIN = out['train_loss_list'][-1]
        d = {'Test': [ACC_TEST], 'Train': [ACC_TRAIN]}

        pd.DataFrame(d).to_csv(os.path.join(event['output_dir'], 'ACC_TESTRAIN.csv'), index=False)

        out_dict['acc_test'] = ACC_TEST
        out_dict['acc_train'] = ACC_TRAIN
        out_dict['confusion_test'] = pd.DataFrame(CONFUSION_TEST)

    # SHAPley values
    if event.get('shapley', False):
        out_dataframes_shap = get_shapley(event, joint_model, datas)
        out_dict['shapley'] = out_dataframes_shap

    # Integrated gradients
    if event.get('igrads', False):
        out_dataframes_igrad = get_igrads(event, joint_model, datas)
        out_dict['igrads'] = out_dataframes_igrad

    return out_dict

def get_shapley(event, joint_model, datas):
    joint_model_wrp = JointMLPWrapper(joint_model)

    # make the explainer object
    explainer = shap.DeepExplainer(joint_model_wrp, list(datas.values()))

    # make tensors of the whole dataset
    all_data_tensors = [torch.tensor(d.values, dtype=torch.float32).T for d in datas]

    # compute the shapley values
    shap_values = explainer.shap_values(all_data_tensors, check_additivity=False)

    # make all the plots and dump them to disk
    # TODO:  I am assuming binary classification, for multi-class we need to loop over the first index as well probably
    for k, (tmp_tensor, sv, fname) in enumerate(zip(all_data_tensors, shap_values[0], event['edata_filenames'])):
        fig = plt.gcf()
        shap.summary_plot(sv, features=tmp_tensor, feature_names=datas[k].index, show=False)
        fig.savefig(os.path.join(event['output_dir'], 'shap_summary_plot_{}.png'.format(fname)))
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
        tmp_df.to_csv(os.path.join(event['output_dir'], 'shapley_values_{}.csv'.format(fname)), index=False)

    return(out_dataframes_shap)

def get_igrads(event, joint_model, datas):
    # TODO:  Again, here I'm assuming binary classification, we'd need to examine each output dimension for multi-class.

    all_data_tensors = [torch.tensor(d.values, dtype=torch.float32).T for d in datas]

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
        tmp_df.to_csv(os.path.join(event['output_dir'], 'integrated_gradients_{}.csv'.format(fname)), index=False)

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
        fig.savefig(os.path.join(event['output_dir'], 'integrated_gradients_{}.png'.format(edata_cname)), bbox_inches='tight')
        plt.close(fig)

    return out_dataframes_igrad
