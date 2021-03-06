#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
import h5py
import time
import joblib

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn import ensemble, metrics, inspection

def MomentumModulus(px, py, pz):
    return np.sqrt(px**2 + py**2 + pz**2)

def TransverseMomentumModulus(px, py):
    return np.sqrt(px**2 + py**2)

### MONTE CARLO DATA

inputMC = 'data/LHCbMC_2016-2017-2018_MagUpDown_Lb2JPsiL_Ttracks_v12.h5'
tickMC = time.perf_counter()
df_reco = pd.read_hdf(inputMC, key='LHCbMC_Lb')
tockMC = time.perf_counter()
print(f"Monte Carlo imported in {tockMC - tickMC:0.4f} seconds.")

tickMCTruth = time.perf_counter()
df_truth = pd.read_hdf(inputMC, key='LHCbMCTruth_Lb')
tockMCTruth = time.perf_counter()
print(f"Monte Carlo Truth imported in {tockMCTruth - tickMCTruth:0.4f} seconds.")

tickMerge = time.perf_counter()
df_MC = pd.merge(df_truth.loc[df_truth['Rec_key'] >= 0], df_reco, left_index=True, right_on='MC_key')
df_MC = df_MC.loc[(df_MC['MC_key'] >= 0) & (df_MC['Rec_key'] >= 0)]
tockMerge = time.perf_counter()
print(f"Monte Carlo merged in {tockMerge - tickMerge:0.4f} seconds.")

JPsi1SPDGMass = 3096.900

PionPCuts = (MomentumModulus(df_MC['pim_PX'], df_MC['pim_PY'], df_MC['pim_PZ']) > 2000) & (MomentumModulus(df_MC['pim_PX'], df_MC['pim_PY'], df_MC['pim_PZ']) < 5e5)
ProtonPCuts = (MomentumModulus(df_MC['p_PX'], df_MC['p_PY'], df_MC['p_PZ']) > 10000) & (MomentumModulus(df_MC['p_PX'], df_MC['p_PY'], df_MC['p_PZ']) < 5e5)
ProtonPTCuts = TransverseMomentumModulus(df_MC['p_PX'], df_MC['p_PY']) > 400
## Combined m(p-pi)? Seems to be "AM" in the DaVinci opt file
LambdaMCuts = (df_MC['L_M'] > 600) & (df_MC['L_M'] < 1500)
LambdaMMCuts = df_MC['L_MM'] < 1500
LambdaZCuts = (df_MC['L_ENDVERTEX_Z'] > 5500) & (df_MC['L_ENDVERTEX_Z'] < 8500)
LambdaDiraCuts = (df_MC['L_BPVDIRA'] > 0.9999)
LambdaBPVIPCHI2Cuts = df_MC['L_BPVIPCHI2'] < 200
LambdaBPVVDCHI2Cuts = df_MC['L_BPVVDCHI2'] < 2e7
LambdaChi2Cuts = df_MC['L_VFASPF_CHI2_VDOF'] < 750
JPsiMCuts = abs(df_MC['Jpsi_M'] - JPsi1SPDGMass) < 90
LambdaPTCuts = TransverseMomentumModulus(df_MC['L_PX'], df_MC['L_PY']) > 450
## Combined m(JpsiLambda)? See comment above
LambdabMCuts = (df_MC['Lb_M'] < 8500)
LambdabDiraCuts = abs(df_MC['Lb_BPVDIRA']) > 0.99
LambdabBPVIPCHI2Cuts = df_MC['Lb_BPVIPCHI2'] < 1750
LambdabChi2Cuts = df_MC['Lb_VFASPF_CHI2_VDOF'] < 150

df_MC_Filtered = df_MC.loc[
	PionPCuts &
	ProtonPCuts &
	ProtonPTCuts &
	LambdaMCuts &
	LambdaMMCuts &
	LambdaZCuts &
	LambdaDiraCuts &
	LambdaBPVIPCHI2Cuts &
	LambdaBPVVDCHI2Cuts &
	LambdaChi2Cuts &
	JPsiMCuts &
	LambdaPTCuts &
	LambdabMCuts &
	LambdabDiraCuts &
	LambdabBPVIPCHI2Cuts &
	LambdabChi2Cuts
]

### REAL DATA

inputData = 'data/Custom_Shuffled5e5_LHCbData_2016_MagUpDown_Dimuon_Ttracks.h5'

tickData = time.perf_counter()
df_Data = pd.read_hdf(inputData, key='LHCbData')
tockData = time.perf_counter()
print(f"Data imported found in {tockData - tickData:0.4f} seconds.")

df_Data_Sideband = df_Data.loc[df_Data['DTF_FixJPsiLambda_Lb_M'] > 5803]

### ADD MISSING FEATURES

df_MC_Filtered = df_MC_Filtered.assign(
	p_PT = TransverseMomentumModulus(df_MC_Filtered['p_PX'],df_MC_Filtered['p_PY']),
	DTF_FixJPsiLambda_p_PT = TransverseMomentumModulus(df_MC_Filtered['DTF_FixJPsiLambda_p_PX'], df_MC_Filtered['DTF_FixJPsiLambda_p_PY']),
	pim_PT = TransverseMomentumModulus(df_MC_Filtered['pim_PX'],df_MC_Filtered['pim_PY']),
	DTF_FixJPsiLambda_pim_PT = TransverseMomentumModulus(df_MC_Filtered['DTF_FixJPsiLambda_pim_PX'], df_MC_Filtered['DTF_FixJPsiLambda_pim_PY']),
	Jpsi_PT = TransverseMomentumModulus(df_MC_Filtered['Jpsi_PX'],df_MC_Filtered['Jpsi_PY'])
)

successDictionaryReverse = {
	'Success': 0.0,
	'Failed': 1.0,
	'NonConverged': 3.0
}

df_MC_Filtered.replace({'DTF_FixJPsi_status': successDictionaryReverse}, inplace=True)
df_MC_Filtered.replace({'DTF_FixJPsiLambda_status': successDictionaryReverse}, inplace=True)

df_Data_Sideband = df_Data_Sideband.assign(
	p_PT = TransverseMomentumModulus(df_Data_Sideband['p_PX'],df_Data_Sideband['p_PY']),
	DTF_FixJPsiLambda_p_PT = TransverseMomentumModulus(df_Data_Sideband['DTF_FixJPsiLambda_p_PX'], df_Data_Sideband['DTF_FixJPsiLambda_p_PY']),
	pim_PT = TransverseMomentumModulus(df_Data_Sideband['pim_PX'],df_Data_Sideband['pim_PY']),
	DTF_FixJPsiLambda_pim_PT = TransverseMomentumModulus(df_Data_Sideband['DTF_FixJPsiLambda_pim_PX'], df_Data_Sideband['DTF_FixJPsiLambda_pim_PY']),
	Jpsi_PT = TransverseMomentumModulus(df_Data_Sideband['Jpsi_PX'],df_Data_Sideband['Jpsi_PY'])
)

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

## SELECT TRAINING DATA

df_signal = df_MC_Filtered.sample(frac=1, random_state=98)
df_signal['TYPE'] = 1
df_background  = df_Data_Sideband.sample(frac=1, random_state=98)
df_background['TYPE'] = 0
signalSplitPoint = int(len(df_signal) * 0.9)
backgroundSplitPoint = int(len(df_background) * 0.9)

df_train = pd.concat([df_signal.iloc[:signalSplitPoint], df_background.iloc[:backgroundSplitPoint]], ignore_index=True)
df_train.dropna('columns', inplace=True)

X_train = df_train.loc[:, features].to_numpy()
Y_train = df_train.loc[:, 'TYPE'].to_numpy()

## FINALLY, TRAIN THE HBDT

HBDT = ensemble.HistGradientBoostingClassifier(
	random_state=2021,
	learning_rate=0.01,
	max_leaf_nodes=100,
	max_iter=2500,
)

print("Begin the training...")
tickHBDT = time.perf_counter()
HBDT.fit(X_train, Y_train)
tockHBDT = time.perf_counter()
print(f"BDT trained in {(tockHBDT - tickHBDT)/3600:0.4f} hours.")
tickSave = time.perf_counter()
joblib.dump(HBDT, 'HBDT.joblib')
tockSave = time.perf_counter()
print(f"BDT saved in {(tockSave - tickSave)/60:0.4f} minutes.")
