import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from diomics.models.multi_mlp import make_joint_model
from diomics.featimportance.igrads import integrated_grads
from diomics.featimportance.shapley import JointMLPWrapper

import shap
import numpy as np
import pandas as pd
import copy
import datetime
import logging
import os
import re
import json
import pickle
import tempfile

from sklearn.metrics import roc_auc_score, confusion_matrix, v_measure_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Dummy event object.  Passed as the 'data' field in curl

# args['fmeta_sample_names'] = ['SampleID_proteomics', 'SampleID_lipidpos', 'SampleID_metab']
# args['fmeta_target_name'] = "Virus"
# args['edata_cnames'] = ['Metabolite', 'Name', "Protein"]

# event = {
#     "do_train": "true",
#     "do_eval": "true",
#     "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
#     "fmeta_target_name": "Virus",
#     "train_data_paths": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
#     "edata_cnames": ["Metabolite", "Name", "Protein"],
#     "shapley": "true",
#     "igrads": "true"
# }

def train_joint_model(
    joint_model, train_tensors, y_gt, optimizer, args, do_validation = True, 
    valid_tensors = None, y_gt_valid = None, max_iter = 5000, mlflow_stage = "tuning",
    reduce_lr = False):

    best_valid_loss = torch.tensor(torch.inf)
    iter_no_improvement = 0

    valid_loss_list = []
    train_loss_list = []
    pbar = tqdm(range(max_iter))
    valid_loss = torch.tensor(torch.inf)
    best_iter = 0

    if args['marginal_coefs'] is None:
        args['marginal_coefs'] = [1.0 for _ in range(len(train_tensors))]

    # create validation tensors
    if do_validation and valid_tensors is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1565)
        train_idx, valid_idx = next(splitter.split(train_tensors[0], y_gt))

        valid_tensors = [el[valid_idx] for el in train_tensors]
        y_gt_valid = y_gt[valid_idx]

        train_tensors = [el[train_idx] for el in train_tensors]
        y_gt = y_gt[train_idx]

    for i in pbar:
        yhat, h, yhats, hiddens = joint_model(*train_tensors)

        product_loss, marginal_losses, loss = joint_model.loss(
            y_gt, yhat, yhats, alpha=args['alpha'], gamma=args['gamma'],
            marginal_coefs=args['marginal_coefs'], marginal_weight = args['marginal_weight']
        )

        optimizer.zero_grad()

        if args['marginal_loss']:
            loss.backward()
        else:
            loss = product_loss
            loss.backward()

        # torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 2.0)
        optimizer.step()

        # validation step
        if do_validation and i % args['validation_frequency'] == 0 and i > 0:
            assert valid_tensors is not None and y_gt_valid is not None, "Validation tensors and ground truth must be provided if do_validation is True"
            joint_model.eval()

            yhat_valid, h_valid, yhats_valid, hiddens_valid = joint_model(*valid_tensors)
            _, _, valid_loss = joint_model.loss(
                y_gt_valid, yhat_valid, yhats_valid,
                alpha=args['alpha'], gamma=args['gamma'],
                marginal_coefs=args['marginal_coefs'], marginal_weight = args['marginal_weight']    
            )
            valid_loss_list.append(valid_loss.item())
            mlflow.log_metric(f"{mlflow_stage}_validation_loss", valid_loss.item(), step=i)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_iter = i
                print(f'Validation loss improved to {valid_loss.item()}, saving model')
                torch.save(joint_model.state_dict(), args['best_model_path'])
                iter_no_improvement = 0
            else:
                iter_no_improvement += 1

            if iter_no_improvement > args['patience']:
                if reduce_lr:
                    print(f'No improvement in validation loss for {int(args["patience"] * args["validation_frequency"])} iterations, reducing learning rate')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.25
                    reduce_lr = False
                    iter_no_improvement = 0
                else:
                    print(f'No improvement in validation loss for {int(args["patience"] * args["validation_frequency"])} iterations, stopping training')
                    break

            joint_model.train()

        # set the tqdm message:
        pbar.set_description(f'Iteration {i}, training loss: {loss.item():.5f} validation loss: {valid_loss.item():.5f} best validation loss: {best_valid_loss.item():.5f}')

        train_loss_list.append(loss.item())
        mlflow.log_metric(f"{mlflow_stage}_training_loss", loss.item(), step=i)

    out = {
        "joint_model": joint_model,
        "valid_loss_list": valid_loss_list,
        "train_loss_list": train_loss_list,
        "n_iter": i,
        "best_iter": best_iter + 1
    }

    return out

def run_experiment(args):
    logging.info("arguments: %s", args)

    os.makedirs(args['output_dir'], exist_ok=True)

    # flags to do training and/or evaluation
    do_train = args.get('do_train', False)
    do_eval = args.get('do_eval', False)
    
    # load data
    train_X_list = []

    for path in args['train_data_paths']:
        path = os.path.join(args['data_dir'], path)

        if os.path.splitext(path)[1] == '.csv':
            train_dat = pd.read_csv(path)
        elif os.path.splitext(path)[1] == '.xlsx':
            train_dat = pd.read_excel(path)
        else:
            raise ValueError("Unknown file type: {}, must be one of .csv or .xlsx".format(path))
    
        train_X = train_dat.set_index("SampleID").drop("y", axis = 1)

        train_X = train_X - train_X.median().median()

        train_X_list.append(train_X)
        
    train_y = train_dat['y']

    test_X_list = []

    for path in args['test_data_paths']:
        path = os.path.join(args['data_dir'], path)

        if os.path.splitext(path)[1] == '.csv':
            test_dat = pd.read_csv(path)
        elif os.path.splitext(path)[1] == '.xlsx':
            test_dat = pd.read_excel(path)
        else:
            raise ValueError("Unknown file type: {}, must be one of .csv or .xlsx".format(path))

        test_X = test_dat.set_index("SampleID").drop("y", axis = 1)

        test_X = test_X - test_X.median().median()

        test_X_list.append(test_X)

    # construct the target, everything should be aligned
    test_y = test_dat['y']

    y_gt = train_y.astype('category').cat.codes
    y_gt = torch.tensor(y_gt.values, dtype=torch.int64)

    ytest_gt = test_y.astype('category').cat.codes
    ytest_gt = torch.tensor(ytest_gt.values, dtype=torch.int64)

    cat_names = train_y.astype('category').cat.categories

    # Load the time data
    for i, path in enumerate(args['fdata_paths']):
        path = os.path.join(args['data_dir'], path)

        if os.path.splitext(path)[1] == '.csv':
            fdata = pd.read_csv(path)
        else:
            raise ValueError("Unknown file type: {}, must be one of [.csv]".format(path))

        train_time = fdata[fdata.in_train == 1]['Time']
        test_time = fdata[fdata.in_train == 0]['Time']

        train_time.index = fdata[fdata.in_train == 1]['SampleID']
        test_time.index = fdata[fdata.in_train == 0]['SampleID']

        train_X_list[i]['Time'] = train_time
        test_X_list[i]['Time'] = test_time

    # make the joint model, need the arguments for each marginal model
    joint_model = make_joint_model(
        train_X_list, 
        prediction_dim = train_y.nunique(),
        hidden_sizes = args['hidden_sizes'],
        dropout = args['dropout'],
        hidden_dim = args['hidden_dim'],
        activation_fn = args['activation_fn'],
        combine_fn = args['combine_fn']
    )

    # define optimizer
    optimizer = AdamW(joint_model.parameters(), lr=1e-4)

    train_tensors = [torch.tensor(d.values, dtype=torch.float32) for d in train_X_list]
    test_tensors = [torch.tensor(d.values, dtype=torch.float32) for d in test_X_list]

    if do_train:
        out = train_joint_model(
            joint_model, train_tensors, y_gt, optimizer, args, do_validation = True,
            valid_tensors = None, y_gt_valid = None, max_iter = args['max_iter']
        )

        # train on the training + validation data
        joint_model = make_joint_model(
            train_X_list, 
            prediction_dim = train_y.nunique(), 
            hidden_sizes = args['hidden_sizes'], 
            dropout = args['dropout'], 
            hidden_dim = args['hidden_dim'],
            activation_fn = args['activation_fn'],
            combine_fn = args['combine_fn']
        )

        optimizer = AdamW(joint_model.parameters(), lr=5e-5)

        out = train_joint_model(
            joint_model, train_tensors, y_gt, optimizer, args,
            max_iter = int(out['best_iter'] / (1 - 0.3)), 
            do_validation = False, mlflow_stage = "final"
        )

        joint_model = out['joint_model']

        # save the pytorch model
        if args['final_model_path'] is None:
            # save to a temp path
            with tempfile.TemporaryDirectory() as tempdir:
                os.makedirs(tempdir, exist_ok=True)
                model_path = os.path.join(tempdir, "final_model.pt")
            
                torch.save(joint_model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
        else:
            torch.save(joint_model.state_dict(), args['final_model_path'])
            mlflow.log_artifact(args['final_model_path'])
    else:
        # load the pytorch model
        joint_model.load_state_dict(torch.load(args['best_model_path']))
    
    joint_model.eval()

    if do_eval:
        with torch.inference_mode():
            yhat, poe_dist, yhats, dists = joint_model(*test_tensors)

        ypred = yhat.argmax(dim=1).numpy()

        ACC_TEST = (ypred == ytest_gt.numpy()).mean()

        # get roc_score
        if yhat.shape[1] > 2:
            AUC_TEST = roc_auc_score(ytest_gt, yhat.numpy(), multi_class='ovr')
        elif yhat.shape[1] == 2:
            AUC_TEST = roc_auc_score(ytest_gt, yhat[:, 1].numpy())
        else:
            logging.warning("AUC not computed, only one class in the output...")
            AUC_TEST = None

        mlflow.log_metric("roc_auc", AUC_TEST)

        CONFUSION_TEST = confusion_matrix(ytest_gt, ypred)

        # dump eval results, just one line with the accuracy for now
        with open(os.path.join(args['output_dir'], 'eval_results.txt'), 'w') as f:
            f.write(f"Model test accuracy: {ACC_TEST}\n")
        mlflow.log_metric("test_accuracy", ACC_TEST)

        # dump the confusion matrix
        confusion_csv_path = os.path.join(args['output_dir'], 'confusion_matrix.csv')
        pd.DataFrame(CONFUSION_TEST).to_csv(confusion_csv_path, index=False)
        mlflow.log_artifact(confusion_csv_path)

        
    # SHAPley values
    if args.get('shapley', False):
        all_data_tensors = [torch.tensor(d.values, dtype=torch.float32) for d in train_X_list]
        shap_values = run_shapley(joint_model, train_tensors, train_X_list, cat_names, args)

    # Integrated gradients
    # TODO:  Again, here I'm assuming binary classification, we'd need to examine each output dimension for multi-class.
    if args.get('igrads', False):
        all_data_tensors = [torch.tensor(d.values, dtype=torch.float32) for d in train_X_list]
        baselines = [-torch.rand_like(el)/2 for el in all_data_tensors]
        # baselines = [el[:, torch.randperm(el.shape[1])] for el in baselines]

        all_igrad_scores = []

        for class_idx, cat_name in enumerate(cat_names):
            tmp_igrad_scores = integrated_grads(all_data_tensors, baselines, joint_model, n_steps=100, class_idx=class_idx)

            all_igrad_scores.append(tmp_igrad_scores)

            abs_igrad_scores = [el.abs().sum(axis=0).numpy() for el in tmp_igrad_scores]
            sorted_igrad_scores = [sorted(el, reverse=True) for el in abs_igrad_scores]

            out_dataframes_igrad = []

            for i, fname in enumerate(args['train_data_paths']):
                tmp_df = pd.DataFrame({
                    'feature': train_X_list[i].iloc[:,(-abs_igrad_scores[i]).argsort()].columns, 
                    'IGrad_score': sorted_igrad_scores[i]
                })

                scores_df = pd.DataFrame(tmp_igrad_scores[i].T, columns=train_X_list[i].index)
                scores_df.index = train_X_list[i].columns

                tmp_df = tmp_df.merge(scores_df, how = 'left', left_on='feature', right_index=True)
                out_dataframes_igrad.append(tmp_df)

                igrad_path = os.path.join(args['output_dir'], f'integrated_gradients_{os.path.basename(fname)}_{class_idx}.csv')
                tmp_df.to_csv(igrad_path, index=False)
                mlflow.log_artifact(igrad_path)

        igrads_path = os.path.join(args['output_dir'], f'igrad_scores.pkl')
        pickle.dump(all_igrad_scores, open(igrads_path, 'wb'))
        mlflow.log_artifact(igrads_path)

    if args.get('eval_clustering', False) or args.get('eval_prediction', False):
        subsets_by_idx = {}
        subsets_by_idx_test = {}

        if args.get('shapley', False):
            for prop in args['top_n']:
                for i, cat_name in enumerate(cat_names):
                    sset = take_top_tensors(all_data_tensors, shap_values[i], top_n=prop, per_view=False)
                    subsets_by_idx[f"shapley_{cat_name}_{prop}"] = sset

                    sset_test = take_top_tensors(test_tensors, shap_values[i], top_n=prop, per_view=False)
                    subsets_by_idx_test[f"shapley_{cat_name}_{prop}"] = sset_test

        if args.get('igrads', False):
            for prop in args['top_n']:
                for i, cat_name in enumerate(cat_names):
                    sset = take_top_tensors(all_data_tensors, all_igrad_scores[i], top_n=prop, per_view=False)
                    subsets_by_idx[f"igrads_{cat_name}_{prop}"] = sset

                    sset_test = take_top_tensors(test_tensors, all_igrad_scores[i], top_n=prop, per_view=False)
                    subsets_by_idx_test[f"igrads_{cat_name}_{prop}"] = sset_test

    if args.get("eval_clustering", False):
        if args.get('shapley', False) or args.get('igrads', False):
            for k, v in subsets_by_idx.items():
                X = np.concatenate([el.numpy() for el in v], axis=1)
                kmeans = KMeans(n_clusters=len(cat_names), random_state=0).fit(X)
                v_measure = v_measure_score(y_gt.numpy(), kmeans.labels_)
                mlflow.log_metric(f"v_measure_{k}_kmeans", v_measure)

                kmeans_path = os.path.join(args['output_dir'], f'kmeans_{k}.pkl')
                with open(kmeans_path, 'wb') as f:
                    pickle.dump(kmeans, f)

                mlflow.log_artifact(kmeans_path)

                agglom = AgglomerativeClustering(n_clusters=len(cat_names)).fit(X)
                v_measure = v_measure_score(y_gt.numpy(), agglom.labels_)
                mlflow.log_metric(f"v_measure_{k}_agglom", v_measure)

                agglom_path = os.path.join(args['output_dir'], f'agglom_{k}.pkl')
                with open(agglom_path, 'wb') as f:
                    pickle.dump(agglom, f)

                mlflow.log_artifact(agglom_path)

    if args.get("eval_prediction", False):
        if args.get('shapley') | args.get('igrads'):
            for k, v in subsets_by_idx.items():
                # train a classifier on the subsetted data
                X = np.concatenate([el.numpy() for el in v], axis=1)
                X_test = np.concatenate([el.numpy() for el in subsets_by_idx_test[k]], axis=1)
                
                rf = RandomForestClassifier(random_state=0)
                clf = GridSearchCV(
                    rf, 
                    {
                        'max_depth': [4, 6, 8, None],
                        'n_estimators': [10, 100, 150]
                    }, 
                    cv=5
                )

                logging.info(f"Training classifier on {k}")
                clf.fit(X, y_gt.numpy())
                
                pickle.dump(clf, open(os.path.join(args['output_dir'], f'rf_gridsearch_{k}.pkl'), 'wb'))
                mlflow.log_artifact(os.path.join(args['output_dir'], f'rf_gridsearch_{k}.pkl'))

                clf = clf.best_estimator_
                ypred = clf.predict(X_test)
                yprob = clf.predict_proba(X_test)

                ACC_TEST = (ypred == ytest_gt.numpy()).mean()
                CONFUSION_TEST = confusion_matrix(ytest_gt, ypred)

                if yprob.shape[1] > 2:
                    AUC_TEST = roc_auc_score(ytest_gt, yprob, multi_class='ovr')
                else:
                    AUC_TEST = roc_auc_score(ytest_gt, yprob[:, 1])

                mlflow.log_metric(f"test_accuracy_RF_{k}", ACC_TEST)
                mlflow.log_metric(f"test_auc_RF_{k}", AUC_TEST)

                confusion_csv_path = os.path.join(args['output_dir'], f'confusion_matrix_{k}.csv')
                pd.DataFrame(CONFUSION_TEST).to_csv(confusion_csv_path, index=False)
                mlflow.log_artifact(confusion_csv_path)

    return("success")

def run_shapley(joint_model, train_tensors, train_X_list, cat_names, args):
    joint_model_wrapper = JointMLPWrapper(joint_model)

    # make the explainer object
    explainer = shap.DeepExplainer(joint_model_wrapper, train_tensors)

    # make tensors of the whole dataset
    all_data_tensors = [torch.tensor(d.values, dtype=torch.float32) for d in train_X_list]

    # compute the shapley values
    shap_values = explainer.shap_values(all_data_tensors, check_additivity=False)

    pickle_path = os.path.join(args['output_dir'], 'shapley_values.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(shap_values, f)
    
    mlflow.log_artifact(pickle_path)

    for i, cat_name in enumerate(cat_names):
        abs_shap_values = [np.abs(el).mean(axis = 0) for el in shap_values[i]]
        sorted_shap_values = [sorted(el, reverse=True) for el in abs_shap_values]

        out_dataframes_shap = []

        for j, fname in enumerate(args['train_data_paths']):
            tmp_df = pd.DataFrame({
                'feature': train_X_list[j].iloc[:, (-abs_shap_values[j]).argsort()].columns, 
                'shapley_value': sorted_shap_values[j]
            })

            scores_df = pd.DataFrame(shap_values[i][j].T, columns=train_X_list[j].index)
            scores_df.index = train_X_list[j].columns

            tmp_df = tmp_df.merge(scores_df, how = 'left', left_on='feature', right_index=True)
            out_dataframes_shap.append(tmp_df)

            shap_csv_path = os.path.join(args['output_dir'], 'shapley_values_{}_{}.csv'.format(os.path.basename(fname), cat_name))

            tmp_df.to_csv(shap_csv_path, index=False)
            
            mlflow.log_artifact(shap_csv_path)

    return shap_values
            

def take_top_tensors(tensors, scores, top_n = 0.1, per_view = True):
    """Subset tensors based on attribution scores.

    Args:
        tensors (List[torch.tensor]): a list of n x p tensors of the raw input data to be used for clustering.
        scores (List[array]): A list of scores for each dataset indicating their relative importance or attribution scores.
        top_n (Any): Any one of: An integer or float between (0,1) specifying the number or proportion of features to subset to for each data tensor.  A list of integers and/or floats specifying the per-dataset number or proportion of features to subset to.
    """
    tensors = tensors.copy()

    # the scores will be already subsetted to the 'first' dimension, in the case of multi-class classification, we need to loop over the classes

    abs_scores = {
        i: np.abs(scores[i]).mean(axis=0) for i in range(len(scores))
    }

    if per_view == True:
        score_ranks = {}
        for k, v in abs_scores.items():
            score_ranks[k] = np.argsort(v)[::-1]
        
        if isinstance(top_n, int):
            top_n = [top_n for _ in range(len(tensors))]
        elif isinstance(top_n, float):
            top_n = [int(top_n * t.shape[1]) for t in tensors]

        # subset the features to the top n
        for i, (t, rank, n), in enumerate(zip(tensors, score_ranks.values(), top_n)):
            if isinstance(top_n, int):
                top_n = [top_n for _ in range(len(rank))]

            tensors[i] = t[:, rank[:n].copy()]
    else:
        assert not isinstance(top_n, list), "If per_view is False, top_n must be a single value"

        score_ranks = np.argsort(np.concatenate([v for v in abs_scores.values()]))[::-1]

        if isinstance(top_n, int):
            score_ranks = score_ranks[:top_n]
        elif isinstance(top_n, float):
            score_ranks = score_ranks[:int(top_n * len(score_ranks))]

        begin_idx = 0
        concat_tensors = torch.cat(tensors, dim=1)

        for i, t in enumerate(tensors):
            index_range = range(begin_idx, begin_idx + t.shape[1])
            overlap_inds = [el for el in index_range if el in score_ranks]
            tensors[i] = concat_tensors[:, overlap_inds]

            begin_idx += t.shape[1]

    return tensors

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg : DictConfig) -> None:
    # import pdb;pdb.set_trace()
    args = OmegaConf.to_container(cfg)

    args['activation_fn'] = hydra.utils.instantiate(args['activation_fn'])

    # mlflow.set_tracking_uri("file:///Users/clab683/git_repos/DeepIMV/mlruns")
    exp_id = mlflow.set_experiment(experiment_name=args['experiment_name'])

    print(mlflow.get_tracking_uri())

    with mlflow.start_run():
        mlflow.set_tags({'combine_fn': args['combine_fn']})
        mlflow.log_params(args)

        run_experiment(args)

if __name__ == "__main__":
    main()
    