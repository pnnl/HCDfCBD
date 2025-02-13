import torch
from torch.optim import AdamW
from diomics.models.multi_mlp import simple_FC, JointMLP
from diomics.models.fc_imv import FC_Marginal, JointVAE
from diomics.featimportance.shapley import JointMLPWrapper
from diomics.featimportance.igrads import average_igrads
from diomics.featimportance.comparison import overlap_sets
import shap
import numpy as np
import pandas as pd

from tqdm import tqdm
import datetime
import tempfile
import pickle

from sklearn.metrics import roc_auc_score
from scipy.stats import weightedtau

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_icl102_multiclass():
    train_lip = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/lip_neg_train.csv")
    test_lip = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/lip_neg_test.csv")

    train_metab = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/metab_train.csv")
    test_metab = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/metab_test.csv")

    train_pro = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/pro_train.csv")
    test_pro = pd.read_csv("/Users/clab683/git_repos/DeepIMV/data/ICL104-MAR2024/pro_test.csv")

    ytrain = train_lip['y']
    ytest = test_lip['y']

    train_lip = train_lip.set_index("SampleID").drop("y", axis=1)
    test_lip = test_lip.set_index("SampleID").drop("y", axis=1)

    train_metab = train_metab.set_index("SampleID").drop("y", axis=1)
    test_metab = test_metab.set_index("SampleID").drop("y", axis=1)

    train_pro = train_pro.set_index("SampleID").drop("y", axis=1)
    test_pro = test_pro.set_index("SampleID").drop("y", axis=1)

    test_lip = test_lip - test_lip.median().median()
    train_lip = train_lip - train_lip.median().median()

    test_metab = test_metab - test_metab.median().median()
    train_metab = train_metab - train_metab.median().median()

    test_pro = test_pro - test_pro.median().median()
    train_pro = train_pro - train_pro.median().median()

    out_dict = {
        'train_metab': train_metab,
        'train_pro': train_pro,
        'train_lip': train_lip,
        'test_metab': test_metab,
        'test_pro': test_pro,
        'test_lip': test_lip,
        'ytrain': ytrain,
        'ytest': ytest
    }

    return out_dict

def load_icl104_binary():
    # proData = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/pro_edata_mlready.csv')
    # lipData = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/lip_edata_mlready.csv')
    # metabData = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/metab_edata_mlready.csv')

    # pro_fdata = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/pro_fdata_mlready.csv')
    # lip_fdata = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/lip_fdata_mlready.csv')
    # metab_fdata = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/metab_fdata_mlready.csv')

    train_lip = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/lip_neg_train.csv')
    test_lip = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/lip_neg_test.csv')

    train_metab = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/metab_train.csv')
    test_metab = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/metab_test.csv')

    train_pro = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/pro_train.csv')
    test_pro = pd.read_csv('/Users/clab683/git_repos/DeepIMV/data/ICL104-binary/pro_test.csv')

    ytrain = train_lip['y']
    ytest = test_lip['y']

    train_lip = train_lip.set_index("SampleID").drop("y", axis=1)
    test_lip = test_lip.set_index("SampleID").drop("y", axis=1)

    train_metab = train_metab.set_index("SampleID").drop("y", axis=1)
    test_metab = test_metab.set_index("SampleID").drop("y", axis=1)

    train_pro = train_pro.set_index("SampleID").drop("y", axis=1)
    test_pro = test_pro.set_index("SampleID").drop("y", axis=1)

    test_lip = test_lip - test_lip.median().median()
    train_lip = train_lip - train_lip.median().median()

    test_metab = test_metab - test_metab.median().median()
    train_metab = train_metab - train_metab.median().median()

    test_pro = test_pro - test_pro.median().median()
    train_pro = train_pro - train_pro.median().median()

    out_dict = {
        'train_metab': train_metab,
        'train_pro': train_pro,
        'train_lip': train_lip,
        'test_metab': test_metab,
        'test_pro': test_pro,
        'test_lip': test_lip,
        'ytrain': ytrain,
        'ytest': ytest
    }

    return out_dict

def train(joint_model, optimizer, views_orig, y_gt, n_iters = 1000, alpha = None, gamma = 3, prog = True, l1_penalty=None):
    train_acc = []
    test_acc = []

    if prog:
        pbar = tqdm(range(n_iters))
    else:
        pbar = range(n_iters)

    for i in pbar:
        # update the mlp
        yhat, h, yhats, hiddens = joint_model(*views_orig)

        # pass the predictions and distributions to the loss function and update parameters
        if type(joint_model) is JointVAE:
            product_loss, marginal_losses, loss = joint_model.loss(y_gt, yhat, h, yhats, hiddens, alpha = alpha, gamma = gamma)
        elif type(joint_model) is JointMLP:
            product_loss, marginal_losses, loss = joint_model.loss(y_gt, yhat, yhats, alpha = alpha, gamma = gamma)

        if l1_penalty is not None:
            l1_loss = l1_penalty * sum([p.abs().sum() for m in joint_model.margin_models for p in m.fc1.parameters()])
            loss_penalized = loss + l1_loss

            optimizer.zero_grad()
            loss_penalized.backward()
        else:
            optimizer.zero_grad()
            loss.backward()

        optimizer.step()

        train_acc.append((yhat.argmax(dim=1).numpy() == y_gt.numpy()).mean())

        if (i+1) % 10 == 0 and prog:
            # update the tqdm message
            pbar.set_postfix({'train_acc': train_acc[-1], 'loss': loss.item()})
            # print(f'Iteration {i+1} loss: {loss.item():.3f}')

    return joint_model, train_acc, test_acc

def evaluate_overlap(shaps_1, shaps_2, n_features = 10):
    # get the set intersection of the above
    overlaps = set((-shaps_1).argsort()[:n_features]).intersection(set((-shaps_2).argsort()[:n_features]))
    tau, _ = weightedtau(shaps_1, shaps_2)

    return overlaps, tau

def make_joint_model(input_sizes, prediction_dim, hidden_sizes=[[128, 64, 64]]*2, dropout=0.2, **kwargs):
    assert len(input_sizes) == len(hidden_sizes), "Number of input sizes not equal to number of hidden sizes"
    
    marginals = [
        simple_FC(
            input_size = sz,
            hidden_sizes = hs,
            prediction_dim = prediction_dim,
            dropout = dropout
        )
        for sz, hs in zip(input_sizes, hidden_sizes)
    ]

    joint_model = JointMLP(marginal_models=marginals, hidden_dim=64, dropout=dropout, **kwargs)
    optimizer = AdamW(joint_model.parameters(), lr = 1e-4)

    return joint_model, optimizer

def make_deepimv(input_sizes, z_dim, prediction_dim, hidden_sizes=[[128, 64, 64]]*2, dropout=0.2, **kwargs):
    assert len(input_sizes) == len(hidden_sizes), "Number of input sizes not equal to number of hidden sizes"

    marginals = [
        FC_Marginal(
            sz,
            hs,
            z_dim,
            prediction_dim = prediction_dim,
            dropout = dropout
        )
        for sz, hs in zip(input_sizes, hidden_sizes)
    ]

    joint_model = JointVAE(marginals, hidden_dim = 64, dropout = dropout)
    optimizer = AdamW(joint_model.parameters(), lr = 1e-4)

    return joint_model, optimizer

def compare_shap_scores(
    *datas,
    y, 
    test_x,
    test_y,
    n_noise, 
    hidden_sizes, 
    z_dim,
    n_evals_per_noise=10, 
    n_iters=1250,
    l1_penalty=None,
    prog=True,
    dropout=0.2,
    **kwargs):
    
    logger.info(f"Starting run with n_noise = {n_noise}, hidden_sizes = {str(hidden_sizes)}")

    # the first dataset should be the one receiving the augmentation
    mean_1 = datas[0].mean().sample(n_noise, replace = True)
    std_1 = datas[0].std().sample(n_noise, replace = True)

    data_1_noise = np.random.normal(mean_1, std_1, (datas[0].shape[0], n_noise))

    data_1_noise_test = np.random.normal(mean_1, std_1, (test_x[0].shape[0], n_noise))

    test_x_aug = [
        pd.concat([test_x[0], pd.DataFrame(data_1_noise_test, index = test_x[0].index)], axis = 1),
        *test_x[1:]
    ]

    test_tensors = [torch.tensor(t.values, dtype = torch.float32) for t in test_x]
    test_tensors_aug = [torch.tensor(t.values, dtype = torch.float32) for t in test_x_aug]

    data_1_copy = datas[0].copy()
    # data_2_copy = data_2.copy()

    # attach the noise to the copies
    data_1_copy = pd.concat([data_1_copy, pd.DataFrame(data_1_noise, index = data_1_copy.index)], axis = 1)
    # data_2_copy = pd.concat([data_2_copy, pd.DataFrame(data_2_noise, index = data_2_copy.index)], axis = 1)

    tensor_1 = torch.tensor(datas[0].values, dtype = torch.float32)
    tensor_1_aug = torch.tensor(data_1_copy.values, dtype = torch.float32)

    tensors_rest = [torch.tensor(d.values, dtype = torch.float32) for d in datas[1:]]

    views_orig = [tensor_1, *tensors_rest]
    views_aug = [tensor_1_aug, *tensors_rest]
    
    cat_names = y.astype('category').cat.categories
    y_gt = y.astype('category').cat.codes
    y_gt = torch.tensor(y_gt.values, dtype=torch.int64)
    n_cats = y.nunique()

    by_model_results = {}

    for _ in range(n_evals_per_noise):
        if z_dim is not None:
            joint_model_orig, optimizer_orig = make_deepimv([v.shape[1] for v in views_orig], z_dim, n_cats, hidden_sizes, dropout, **kwargs)
            joint_model_aug, optimizer_aug = make_deepimv([v.shape[1] for v in views_aug], z_dim, n_cats, hidden_sizes, dropout, **kwargs)
            joint_model_orig_rand, optimizer_orig_rand = make_deepimv([v.shape[1] for v in views_orig], z_dim, n_cats, hidden_sizes, dropout, **kwargs)
        else:
            joint_model_orig, optimizer_orig = make_joint_model([v.shape[1] for v in views_orig], n_cats, hidden_sizes, dropout, **kwargs)
            joint_model_aug, optimizer_aug = make_joint_model([v.shape[1] for v in views_aug], n_cats, hidden_sizes, dropout, **kwargs)
            joint_model_orig_rand, optimizer_orig_rand = make_joint_model([v.shape[1] for v in views_orig], n_cats, hidden_sizes, dropout, **kwargs)

        joint_model_orig, train_acc, test_acc = train(joint_model_orig, optimizer_orig, views_orig, y_gt, n_iters = n_iters, alpha = None, gamma = 3, l1_penalty=l1_penalty, prog=prog)
        joint_model_aug, train_acc, test_acc = train(joint_model_aug, optimizer_aug, views_aug, y_gt, n_iters = n_iters, alpha = None, gamma = 3, l1_penalty=l1_penalty, prog=prog)
        joint_model_orig_rand, train_acc, test_acc = train(joint_model_orig_rand, optimizer_orig_rand, views_orig, y_gt, n_iters = n_iters, alpha = None, gamma = 3, l1_penalty=l1_penalty, prog=prog)

        # eval all models
        joint_model_orig.eval()
        joint_model_aug.eval()
        joint_model_orig_rand.eval()

        # Compute SHAP values
        joint_model_orig_wrp = JointMLPWrapper(joint_model_orig)
        explainer = shap.DeepExplainer(joint_model_orig_wrp, views_orig)
        shap_values_orig = explainer.shap_values(views_orig, check_additivity=False)

        joint_model_aug_wrp = JointMLPWrapper(joint_model_aug)
        explainer = shap.DeepExplainer(joint_model_aug_wrp, views_aug)
        shap_values_aug = explainer.shap_values(views_aug, check_additivity=False)

        joint_model_orig_rand_wrp = JointMLPWrapper(joint_model_orig_rand)
        explainer = shap.DeepExplainer(joint_model_orig_rand_wrp, views_orig)
        shap_values_orig_rand = explainer.shap_values(views_orig, check_additivity=False)

        ### Compute Integrated Gradients
        
        if type(joint_model_orig) is JointMLP:
            igrads_orig = average_igrads(views_orig, joint_model_orig, cat_names, n_steps = 50, n_runs = 10)
            igrads_aug = average_igrads(views_aug, joint_model_aug, cat_names, n_steps = 50, n_runs = 10)
            igrads_orig_rand = average_igrads(views_orig, joint_model_orig_rand, cat_names, n_steps = 50, n_runs = 10)

        # Compile results:
        # The format of objects is *_(orig/noise/random)_(dataset 1 or 2)
        # format for shap output with shap==0.4.3

        # get overlap of shap values
        scores_orig_1 = np.abs(shap_values_orig[0][0]).mean(axis=0)
        scores_aug_1 = np.abs(shap_values_aug[0][0]).mean(axis=0)

        # remove the noise features
        scores_aug_1_trunc = scores_aug_1[:datas[0].shape[1]]
        scores_rand_1 = np.abs(shap_values_orig_rand[0][0]).mean(axis=0)

        # scores for the second dataset
        # we need to combine these across all datasets, this involves concatenating the features after averaging
        scores_orig_2 = np.concatenate([
            np.abs(sv).mean(axis=0)
            for sv in shap_values_orig[0][1:]
        ])

        scores_aug_2 = np.concatenate([
            np.abs(sv).mean(axis=0)
            for sv in shap_values_aug[0][1:]
        ])

        scores_rand_2 = np.concatenate([
            np.abs(sv).mean(axis=0)
            for sv in shap_values_orig_rand[0][1:]
        ])

        scores_orig_all = np.concatenate([scores_orig_1, scores_orig_2])
        scores_aug_all = np.concatenate([scores_aug_1_trunc, scores_aug_2])
        scores_rand_all = np.concatenate([scores_rand_1, scores_rand_2])

        # we expect that the overlap between the scores_orig and scores_aug to be lower than that of scores_orig and scores_rand
        overlap_orig_aug, tau_orig_aug = evaluate_overlap(scores_orig_all, scores_aug_all)
        overlap_orig_rand, tau_orig_rand = evaluate_overlap(scores_orig_all, scores_rand_all)
        overlap_aug_rand, tau_aug_rand = evaluate_overlap(scores_aug_all, scores_rand_all)

        mlflow.log_metrics(
            {
                'shap_tau_orig_aug': tau_orig_aug,
                'shap_tau_orig_rand': tau_orig_rand,
                'shap_tau_aug_rand': tau_aug_rand
            }
        )

        if type(joint_model_orig) is JointMLP:
            # get overlap of igrads
            igrads_orig_1 = np.mean(np.abs(igrads_orig[0][cat_names[0]]), axis=0)
            igrads_aug_1 = np.mean(np.abs(igrads_aug[0][cat_names[0]]), axis=0)
            igrads_aug_1_trunc = igrads_aug_1[:,:datas[0].shape[1]]
            igrads_rand_1 = np.mean(np.abs(igrads_orig_rand[0][cat_names[0]]), axis=0)

            igrads_orig_2 = np.mean(np.abs(igrads_orig[1][cat_names[0]]), axis=0)
            igrads_aug_2 = np.mean(np.abs(igrads_aug[1][cat_names[0]]), axis=0)
            igrads_rand_3 = np.mean(np.abs(igrads_orig_rand[1][cat_names[0]]), axis=0)

            igrad_orig_all = np.concatenate([igrads_orig_1.mean(axis=0), igrads_orig_2.mean(axis=0)])
            igrad_aug_all = np.concatenate([igrads_aug_1_trunc.mean(axis=0), igrads_aug_2.mean(axis=0)])
            igrad_rand_all = np.concatenate([igrads_rand_1.mean(axis=0), igrads_rand_3.mean(axis=0)])

            overlap_igrad_orig_aug, tau_igrad_orig_aug = evaluate_overlap(igrad_orig_all, igrad_aug_all)
            overlap_igrad_orig_rand, tau_igrad_orig_rand = evaluate_overlap(igrad_orig_all, igrad_rand_all)
            overlap_igrad_aug_rand, tau_igrad_aug_rand = evaluate_overlap(igrad_aug_all, igrad_rand_all)

            mlflow.log_metrics(
                {
                    'igrad_tau_orig_aug': tau_igrad_orig_aug,
                    'igrad_tau_orig_rand': tau_igrad_orig_rand,
                    'igrad_tau_aug_rand': tau_igrad_aug_rand
                }
            )

        y_gt_test = test_y.astype('category').cat.codes
        y_gt_test = torch.tensor(y_gt_test.values, dtype=torch.int64)

        # get test ROC for the original and augmented model:

        # original data score
        yhat, _, _, _ = joint_model_orig(*test_tensors)

        logits = yhat.detach().numpy()
        
        if logits.shape[1] == 2:
            logits = logits[:,1]

        auc_orig = roc_auc_score(y_gt_test.numpy(), logits, multi_class = 'ovr')
        
        # augmented data score
        yhat, _, _, _ = joint_model_aug(*test_tensors_aug)

        logits = yhat.detach().numpy()
        
        if logits.shape[1] == 2:
            logits = logits[:,1]

        auc_aug = roc_auc_score(y_gt_test.numpy(), logits, multi_class = 'ovr')

        mlflow.log_metrics(
            {
                'auc_orig': auc_orig,
                'auc_aug': auc_aug
            }
        )

        by_model_results.setdefault('auc_orig', []).append(auc_orig)
        by_model_results.setdefault('auc_aug', []).append(auc_aug)
        by_model_results.setdefault('results_shap_orig_aug', []).append((overlap_orig_aug, tau_orig_aug))
        by_model_results.setdefault('results_shap_orig_rand', []).append((overlap_orig_rand, tau_orig_rand))
        by_model_results.setdefault('results_shap_aug_rand', []).append((overlap_aug_rand, tau_aug_rand))
        by_model_results.setdefault('shap_values_orig', []).append(shap_values_orig)
        by_model_results.setdefault('shap_values_aug', []).append(shap_values_aug)
        by_model_results.setdefault('shap_values_orig_rand', []).append(shap_values_orig_rand)

        if type(joint_model_orig) is JointMLP:
            by_model_results.setdefault('results_igrad_orig_aug', []).append((overlap_igrad_orig_aug, tau_igrad_orig_aug))
            by_model_results.setdefault('results_igrad_orig_rand', []).append((overlap_igrad_orig_rand, tau_igrad_orig_rand))
            by_model_results.setdefault('results_igrad_aug_rand', []).append((overlap_igrad_aug_rand, tau_igrad_aug_rand))
            by_model_results.setdefault('igrads_orig', []).append(igrads_orig)
            by_model_results.setdefault('igrads_aug', []).append(igrads_aug)
            by_model_results.setdefault('igrads_orig_rand', []).append(igrads_orig_rand)
        
    return by_model_results

@hydra.main(config_path="cfg", config_name="config-noise", version_base=None)
def main(cfg: DictConfig) -> None:
    # import pdb;pdb.set_trace()
    exp_id = mlflow.set_experiment(experiment_name=cfg.experiment_name)
    logging.info(f"Experiment ID: {exp_id}")

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg))

        # train/test, metab/pro/lip, ytrain, ytest
        datadict = load_icl104_binary()

        train_x = [datadict['train_' + dset] for dset in cfg.datasets]
        
        test_x = [datadict['test_' + dset] for dset in cfg.datasets]
        test_y = datadict['ytest']

        results = compare_shap_scores(
            *train_x, 
            y = datadict['ytrain'], 
            n_noise=cfg.layer_x_noise.n_noise, 
            hidden_sizes=cfg.layer_x_noise.layer_sizes,
            z_dim=cfg.z_dim,
            n_evals_per_noise=cfg.n_evals_per_noise,
            n_iters=cfg.n_iters,
            combine_fn=cfg.combine_fn,
            test_x = test_x,
            test_y = test_y,
            l1_penalty = cfg.l1_penalty,
            dropout = cfg.dropout,
            prog=cfg.prog
        )

        # dump results to a tempfile and log as artifact:
        with tempfile.NamedTemporaryFile() as tmp:
            pickle.dump(results, tmp)
            tmp.seek(0)
            mlflow.log_artifact(tmp.name)


if __name__ == "__main__":
    main()