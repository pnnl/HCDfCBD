"""
Helper functions to load curated datasets for experiments.  Here it is ICL104 and ICL102.
"""

import pandas as pd
import os

def load_icl102_multiclass(base_path = "~/git_repos/DeepIMV/data/ICL104-MAR2024"):
    train_lip = pd.read_csv(os.path.join(base_path, "lip_neg_train.csv"))
    test_lip = pd.read_csv(os.path.join(base_path, "lip_neg_test.csv"))

    train_metab = pd.read_csv(os.path.join(base_path, "metab_train.csv"))
    test_metab = pd.read_csv(os.path.join(base_path, "metab_test.csv"))

    train_pro = pd.read_csv(os.path.join(base_path, "pro_train.csv"))
    test_pro = pd.read_csv(os.path.join(base_path, "pro_test.csv"))

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

def load_icl104_binary(base_path = "~/git_repos/DeepIMV/data/ICL104-binary"):
    # proData = pd.read_csv(os.path.join(base_path, 'pro_edata_mlready.csv'))
    # lipData = pd.read_csv(os.path.join(base_path, 'lip_edata_mlready.csv'))
    # metabData = pd.read_csv(os.path.join(base_path, 'metab_edata_mlready.csv'))

    # pro_fdata = pd.read_csv(os.path.join(base_path, 'pro_fdata_mlready.csv'))
    # lip_fdata = pd.read_csv(os.path.join(base_path, 'lip_fdata_mlready.csv'))
    # metab_fdata = pd.read_csv(os.path.join(base_path, 'metab_fdata_mlready.csv'))

    train_lip = pd.read_csv(os.path.join(base_path, 'lip_neg_train.csv'))
    test_lip = pd.read_csv(os.path.join(base_path, 'lip_neg_test.csv'))

    train_metab = pd.read_csv(os.path.join(base_path, 'metab_train.csv'))
    test_metab = pd.read_csv(os.path.join(base_path, 'metab_test.csv'))

    train_pro = pd.read_csv(os.path.join(base_path, 'pro_train.csv'))
    test_pro = pd.read_csv(os.path.join(base_path, 'pro_test.csv'))

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