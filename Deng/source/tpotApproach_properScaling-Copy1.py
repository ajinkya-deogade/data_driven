#!/usr/local/bin/python3
import sys
if sys.version_info >= (3, 5):
    from importlib.util import spec_from_file_location
    
import os
from time import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, log_loss
from scipy.stats import randint as sp_randint
from scipy import interp
#from drivendata_validator import DrivenDataValidator
import itertools
from tpot import TPOTRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def pre_process_train_test_data(train, test, label_var, exclude_scaling):
    labels = np.ravel(train[label_var])
    train = pd.get_dummies(train.drop(label_var, axis=1))
    test = pd.get_dummies(test)

    # match test set and training set columns
    to_drop = np.setdiff1d(test.columns, train.columns)
    to_add = np.setdiff1d(train.columns, test.columns)

    test.drop(to_drop, axis=1, inplace=True)
    test = test.assign(**{c: 0 for c in to_add})

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    test_indices = test.index
    train_indices = train.index
    train_test = pd.concat([train, test])
    train_test.sort_values(['year', 'weekofyear'], inplace=True)
    train_test.interpolate(method='linear', inplace=True)

    print("Shapes before transformation")
    print("Train : ", train.shape)
    print("Test : ", test.shape)
    print("Train + Test : ", train_test.shape)

    numeric_vals = train_test.select_dtypes(include=['int64', 'float64'])
    numeric_vals = numeric_vals.loc[:, [x for x in list(numeric_vals.columns.values) if x not in exclude_scaling]]
    scaler = StandardScaler()
    train_test[numeric_vals.columns] = scaler.fit_transform(numeric_vals)

    train = train_test.loc[train_indices, :]
    test = train_test.loc[test_indices, :]

    train[label_var] = labels

    print("Shapes after transformation")
    print("Train : ", train.shape)
    print("Test : ",  test.shape)

    return train, test

if __name__ == "__main__":
    # data directory
    # DATA_DIR = os.path.join('..', 'data')
    DATA_DIR = 'D:\work\git_repos\data_driven\Deng\data'

    ## define data paths
    data_paths = {'train_x': os.path.join(DATA_DIR, 'dengue_features_train.csv'),
                  'train_y': os.path.join(DATA_DIR, 'dengue_labels_train.csv'),
                   'test_x':  os.path.join(DATA_DIR, 'dengue_features_test.csv')}

    # load training data
    X_train = pd.read_csv(data_paths['train_x'])
    y_train = pd.read_csv(data_paths['train_y'])
    X_train.drop(columns='week_start_date', inplace=True)

    # load test data
    X_test = pd.read_csv(data_paths['test_x'])
    X_test.drop(columns='week_start_date', inplace=True)

    # #### The first thing to notice is that each country's surveys have wildly different numbers of columns, so we'll plan on training separate models for each country and combining our predictions for submission at the end.
    # ### Pre-process Data
    print("Shapes before transformation")
    print("Train : ", X_train.shape)
    print("Train Labels : ", y_train.shape)
    print("Test : ", X_test.shape)
    print("Columns : ", X_train.columns)
    train_data = pd.merge(X_train, y_train, on=['city', 'year', 'weekofyear'])
    train_data.index = np.arange(0, train_data.shape[0])
    X_test.index = np.arange(train_data.shape[0]+1, train_data.shape[0]+X_test.shape[0]+1)

    print("Preprocessing Training")
    label_var = 'total_cases'
    exclude_scaling = ['year', 'weekofyear']
    a_train, a_test = pre_process_train_test_data(train_data, X_test, label_var, exclude_scaling)
    X_train = a_train.drop(label_var, axis=1)
    y_train = np.ravel(a_train[label_var])

    ## restructure train data
    all_train_data = {'features': X_train,
                      'labels': y_train}

    ## restructure test data
    all_test_data = {'features': a_test}

    # ### Cross-validation -- Tune Parameters
    X = all_train_data['features'].values.astype(np.float32)
    y = all_train_data['labels'].astype(np.int16)
    X_test = all_test_data['features'].values.astype(np.float32)
    
    tune_params = 1
    if tune_params > 0:
        bestParams = []
        X = all_train_data['features'].values.astype(np.float32)
        y = all_train_data['labels'].astype(np.int16)
        pipeline_optimizer = TPOTRegressor(scoring='neg_mean_absolute_error', cv=5,
                                            periodic_checkpoint_folder='D:\work\git_repos\data_driven\Deng\data\tpot_best_models_100',
                                            n_jobs=20, random_state=42, verbosity=3, memory='auto',
                                            generations=100, max_eval_time_mins=10)
        pipeline_optimizer.fit(X, y)
        pipeline_optimizer.export('D:\work\git_repos\data_driven\Deng\source\tpot_best_model_pipeline_gen100.py')
        
        
        # pipeline_optimizer = TPOTRegressor(scoring='neg_mean_absolute_error', cv=5, 
        #                            periodic_checkpoint_folder='D:\work\git_repos\data_driven\Deng\data\tpot_best_models_500\',
        #                            n_jobs=20, random_state=42, verbosity=3, memory='auto',
        #                            generations=500, max_eval_time_mins=10)
        # pipeline_optimizer.fit(X, y)
        # pipeline_optimizer.export('D:\work\git_repos\data_driven\Deng\source\tpot_best_model_pipeline_gen500.py')
