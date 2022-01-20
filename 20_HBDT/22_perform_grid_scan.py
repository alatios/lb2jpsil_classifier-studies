#!/usr/bin/env python3
## Python3 script to perform a HBDT hyperparameter grid search for purposes of optimization
## Results are saved in a CSV file, see notebook 23 (or n+1, where n is the number of this
## file) for the analysis.

import pandas as pd
import numpy as np
import time
#import joblib
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import ensemble, model_selection

## Results CSV file
output_result_directory = './results/'
output_csv = output_result_directory + 'HBDT_crossval_results.csv'

## Load training data
input_train = '~/classifier-studies/data/trainData.h5'
df_train = pd.read_hdf(input_train, 'LHCb_Train')

## Perform the training using DTF_FixJPsi momenta.
features = [
	'DTF_FixJPsi_p_PT',
	'DTF_FixJPsi_p_PZ',
	'DTF_FixJPsi_pim_PT',
	'DTF_FixJPsi_pim_PZ',
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

X_train = df_train.loc[:, features].to_numpy()
Y_train = df_train.loc[:, 'TYPE'].to_numpy()

parameters = {
	'learning_rate': np.logspace(-2.5, -1.75, 4),
	'max_leaf_nodes': [200, 400, 800],
	'max_iter'    : [1500, 2500, 5000],
}

## For testing, this will take less time and will check if the code has bugs
#parameters = {
#    'learning_rate': [0.01, 0.02],
#	'max_leaf_nodes': [200, 400],
#	'max_iter'    : [1500, 2500],
#}

n_jobs = 10

HBDT = ensemble.HistGradientBoostingClassifier(random_state=2022, early_stopping=False)
grid_search = model_selection.GridSearchCV(
	HBDT,
	parameters,
	scoring='average_precision',
	n_jobs=n_jobs,
	verbose=1000,
    cv=model_selection.StratifiedKFold(n_splits=5),
	return_train_score=True
)

print("And now we begin the grid search...")

tick = time.perf_counter()
grid_search.fit(X_train, Y_train)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(output_csv, index=False)
tock = time.perf_counter()

print(f"Grid searched in {(tock - tick)/3600:0.4f} hours.")
