#!/usr/bin/env python3
## Python3 script to perform a BDT hyperparameter grid search for purposes of optimization
## A complete grid search is a computationally heavy matter. For this reason, this script
## only runs a search with a specific maximum numbers of estimators, saving the results in
## an aptly named CSV file. See next notebook for details on CSV merging and result analysis.

## Imports may be a bit too much, I basically kept them from a previous, larger notebook.
## Might be a good idea to check them later.
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt
import h5py
import time
import joblib
from sklearn import ensemble, metrics, inspection, model_selection

## How many estimators (max)?
n_estimators = 200

## Results CSV file
outputDirectory = './results'
outputCSV = 'BDT_cv_results_' + str(n_estimators) + '.csv'

## Load training data
inputTrain = '~/classifier-studies/data/trainData.h5'
df_train = pd.read_hdf(inputTrain, 'LHCb_Train')

## Perform the training using VF momenta. Using them all would be delirious,
## so I chose the most realistic option.
features = [
	'p_PT',
	'p_PZ',
	'pim_PT',
	'pim_PZ',
	'Jpsi_PT',
	'Jpsi_PZ',
	'L_ENDVERTEX_X',
	'L_ENDVERTEX_Y',
	'L_ENDVERTEX_Z',
	'L_BPVDIRA',
	'Lb_BPVDIRA',
	'L_VFASPF_CHI2_VDOF',
	'Lb_VFASPF_CHI2_VDOF',
	'L_BPVIPCHI2',
	'Lb_BPVIPCHI2',
	'L_BPVVDCHI2',
	'Lb_BPVVDCHI2',
	'DTF_FixJPsi_status',
	'DTF_FixJPsiLambda_status'
]

df_train = pd.concat([df_signal.iloc[:signalSplitPoint], df_background.iloc[:backgroundSplitPoint]], ignore_index=True)
df_train.dropna('columns', inplace=True)

X_train = df_train.loc[:, features].to_numpy()
Y_train = df_train.loc[:, 'TYPE'].to_numpy()

parameters = {
	'max_depth'        : list(range(4, 15, 2)),
	'max_features'     : [3, 6, 9],
	'learning_rate'    : [0.05, 0.1, 0.2],
	'subsample'        : [0.7, 1.0],
	'n_estimators'     : [n_estimators]
} #1000 (done), 50 (done), 200, 500
n_jobs = 10

BDT = ensemble.GradientBoostingClassifier(random_state=2021)   
grid_search = model_selection.GridSearchCV(
	BDT,
	parameters,
	scoring='average_precision',
	n_jobs=n_jobs,
	verbose=1000,
	return_train_score=True
)

print("And now we begin the grid search...")

tickGrid = time.perf_counter()

grid_search.fit(X_train, Y_train)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(outputDirectory + '/' + outputCSV, index=False)

tockGrid = time.perf_counter()
print(f"Grid searched in {(tockMCTruth - tickMCTruth)/3600:0.4f} hours.")
