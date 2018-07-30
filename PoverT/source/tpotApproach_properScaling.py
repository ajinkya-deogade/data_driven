#!/usr/bin/python3.6

import os
import sys
from time import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, log_loss
from scipy.stats import randint as sp_randint
from scipy import interp
import itertools
from tpot import TPOTClassifier
from sklearn.preprocessing import StandardScaler

# data directory
DATA_DIR = os.path.join('..', 'data', 'processed')

## Make Submission DataFrame
def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# ##### Household-level survey data: 
# This is obfuscated data from surveys conducted by The World Bank, focusing on household-level statistics. The data come from three different countries, and are separated into different files for convenience.
# 
# ##### Individual-level survey data: 
# This is obfuscated data from related surveys conducted by The World Bank, only these focus on individual-level statistics. The set of interviewees and countries involved are the same as the household data, as indicated by shared id indices, but this data includes detailed (obfuscated) information about household members.
# 
# ##### Submission format:
# This gives us the filenames and columns of our submission prediction, filled with all 0.5 as a baseline.

## define data paths
data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}

# load training data
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')

# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')


# #### The first thing to notice is that each country's surveys have wildly different numbers of columns, so we'll plan on training separate models for each country and combining our predictions for submission at the end.

# ### Pre-process Data

# In[6]:


def pre_process_train_test_data(train, test):
    labels = np.ravel(train.poor)
    train = pd.get_dummies(train.drop('poor', axis=1))
    test =  pd.get_dummies(test)

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

    print("Shapes before transformation")
    print("Train : ", train.shape)
    print("Test : ", test.shape)
    print("Train + Test : ", train_test.shape)

    numeric_vals = train_test.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    train_test[numeric_vals.columns] = scaler.fit_transform(numeric_vals)

    train = train_test.loc[train_indices, :]
    test  = train_test.loc[test_indices, :]
    
    train['poor'] = labels
    
    print("Shapes after transformation")
    print("Train : ", train.shape)
    print("Test : ",  test.shape)
    
    return train, test

print("Preprocessing Training")
print("Country A")
a_train, a_test = pre_process_train_test_data(a_train, a_test)
aX_train = a_train.drop('poor', axis=1)
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
b_train, b_test = pre_process_train_test_data(b_train, b_test)
bX_train = b_train.drop('poor', axis=1)
by_train = np.ravel(b_train.poor)

print("\nCountry C")
c_train, c_test = pre_process_train_test_data(c_train, c_test)
cX_train = c_train.drop('poor', axis=1)
cy_train = np.ravel(c_train.poor)

# In[7]:

## restructure train data
all_train_data = {'A': {'features': aX_train, 
                    'labels': ay_train},
                  'B': {'features': bX_train,
                        'labels':  by_train}, 
                  'C': {'features': cX_train, 
                        'labels':  cy_train}}

## restructure test data
all_test_data = {'A': {'features': a_test},
                 'B': {'features': b_test},
                 'C': {'features': c_test}}


# ### Cross-validation -- Tune Parameters

# In[8]:


tune_params = 1
if tune_params > 0:
   bestParams = []
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
    for grp in all_train_data:

        X = all_train_data[grp]['features'].values.astype(np.float32)
        y = all_train_data[grp]['labels'].astype(np.int16)

        pipeline_optimizer = TPOTClassifier(scoring='neg_log_loss', cv=cv, 
                                            periodic_checkpoint_folder='../data/tpot_best_models_%s/'%(grp),
                                            n_jobs=16, random_state=42, verbosity=3, memory='auto',
                                            generations=20, max_eval_time_mins=10, early_stop=5)
        pipeline_optimizer.fit(X, y)
        pipeline_optimizer.export('tpot_best_model_pipeline_%s.py'%(grp))


### Training Phase
# In[12]:

# trained_models = {}

# ## Train Model A
# from sklearn.ensemble import GradientBoostingClassifier
# trained_models['A'] = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, 
#                                                  max_features=0.5, min_samples_leaf=14, 
#                                                  min_samples_split=20, n_estimators=100, subsample=0.6)
# trained_models['A'].fit(all_train_data['A']['features'], all_train_data['A']['labels'])

# ## Train Model B
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.linear_model import LogisticRegression
# from copy import copy

# trained_models['B'] = make_pipeline(make_union(FunctionTransformer(copy), 
#                                                FunctionTransformer(copy)),
#                                     LogisticRegression(C=0.01, dual=True, penalty="l2"))

# trained_models['B'].fit(all_train_data['B']['features'], all_train_data['B']['labels'])

# ## Train Model C
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.ensemble import RandomForestClassifier
# from tpot.builtins import StackingEstimator

# trained_models['C'] = make_pipeline(
#     make_union(FunctionTransformer(copy), 
#                StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=False))),
#     RandomForestClassifier(bootstrap=True, criterion="entropy", 
#                            max_features=0.8, min_samples_leaf=6, 
#                            min_samples_split=19, n_estimators=100))

# trained_models['C'].fit(all_train_data['C']['features'], all_train_data['C']['labels'])


# ### Testing Phase

# In[13]:


# # ## Predict
# predictions = {}
# for grp in all_train_data:
#     predictions[grp] = trained_models[grp].predict_proba(all_test_data[grp]['features'])


# ### Validate and Submit

# In[14]:


# # convert preds to data frames
# predictionsDF = {}
# for grp in all_train_data:
#     predictionsDF[grp] = make_country_sub(predictions[grp], all_test_data[grp]['features'], grp)

# submission = []
# submission = pd.concat([predictionsDF['A'], predictionsDF['B'], predictionsDF['C']])

# ## Submission Format
# submission.to_csv('../data/my_submission.csv')

# # no parameters unless we have a read_csv kwargs file
# v = DrivenDataValidator()

# if v.is_valid('../data/submission_format.csv', '../data/my_submission.csv'):
#     print "I am awesome."
# else:
#     print "I am not so cool."

