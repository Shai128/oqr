"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
import pandas as pd
import sys
import helper

sys.path.insert(1, '..')


def GetDataset(name, base_path):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""
    if name=="meps_19":
        df = pd.read_csv(base_path + 'meps_19_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        
    if name=="meps_20":
        df = pd.read_csv(base_path + 'meps_20_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        
    if name=="meps_21":
        df = pd.read_csv(base_path + 'meps_21_reg_fix.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
  
        
    if name=="facebook_1":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_1.csv')        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values        
    
    if name=="facebook_2":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_2.csv')        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values 
        
    if name=="bio":
        #https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
    
    if name=='blog_data':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:,0:280].values
        y = df.iloc[:,-1].values
    
    
    UCI_datasets = ['kin8nm', 'naval']

    if name in UCI_datasets:
        data_dir = 'UCI_Datasets/'
        data = np.loadtxt(base_path+data_dir+name+'.txt')
        X = data[:, :-1]
        y = data[:, -1]
        
    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
    except Exception:
        raise Exception("invalid dataset")
    
    return X, y
        
        

log_transform_datasets = ['facebook_1', 'facebook_2', 'blog_data', 'bio']


def get_scaled_data(args):
    dataset_base_path = "datasets/real_data/"
    dataset_name = args.data
    if 'bio' not in dataset_name:
        need_scaling = 'scaled' in dataset_name
        dataset_name = dataset_name.replace('scaled_', '')
    else:
        need_scaling = False
    x,y = GetDataset(dataset_name, dataset_base_path)
    if args.data in log_transform_datasets or need_scaling:
        y = np.log(y - min(y) + 1)
    data_size = len(x)  # data_size_per_dataset[dataset_name]
    # print("data_size: ", data_size)
    idx = np.random.permutation(len(x))[:data_size]
    x = x[idx]
    y = y[idx]
    y = y.reshape(-1, 1)

    return scale_data_wrapper(x,y,args)

def get_synthetic_data(args, minority_group_uncertainty):
    syn_data_path = helper.syn_data_path
    x = pd.read_csv(syn_data_path + 'syn_x_minority_group_uncertainty=' + str(minority_group_uncertainty) +'.csv') \
        .to_numpy().astype(np.float32)
    y = pd.read_csv(syn_data_path + 'syn_y_minority_group_uncertainty=' + str(minority_group_uncertainty) +'.csv') \
        .to_numpy().astype(np.float32)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    group_feature = 0  # feature index that indicates the group
    n_groups = int(max(x[:, group_feature]).item()) + 1


    test_ratio = args.test_ratio
    y_al = y
    syn_x_train, x_te, syn_y_train, y_te = train_test_split(x,
                                                            y,
                                                            test_size=test_ratio)
    unscaled_x_train = syn_x_train
    unscaled_x_test = x_te

    x_tr, x_va, y_tr, y_va = train_test_split(syn_x_train,
                                              syn_y_train,
                                              test_size=0.2)
    scaler = StandardScaler().fit(x_tr)
    unscaled_x_tr = x_tr
    unscaled_x_va = x_va
    syn_x_train = torch.Tensor(scaler.transform(syn_x_train))
    x_te = torch.Tensor(scaler.transform(x_te))
    x_tr = torch.Tensor(scaler.transform(x_tr))
    x_va = torch.Tensor(scaler.transform(x_va))


    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al,
                              unscaled_x_train=unscaled_x_train, unscaled_x_test=unscaled_x_test,
                              unscaled_x_tr=unscaled_x_tr, unscaled_x_va=unscaled_x_va,
                              n_groups=n_groups, group_feature=group_feature,
                              syn_x_train=syn_x_train, syn_y_train=syn_y_train)
    return out_namespace


def scale_data_wrapper(x,y, args):
    test_ratio = args.test_ratio
    return scale_data(x, y, args.seed, test_ratio)


def scale_data(x,y, seed, test_size=0.1):
    x_train, x_te, y_train, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed)

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y))

    x_train = torch.cat([x_tr, x_va], dim=0)
    y_train = torch.cat([y_tr, y_va], dim=0)
    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al,
                              x_train=x_train, y_train=y_train)

    return out_namespace



