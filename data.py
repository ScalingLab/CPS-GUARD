from __future__ import unicode_literals
from sklearn.preprocessing import MaxAbsScaler, Normalizer, RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer

import pandas as pd
import numpy as np

np.random.seed(101)

names = ['MI_dir_L5_weight','MI_dir_L5_mean','MI_dir_L5_variance','MI_dir_L3_weight','MI_dir_L3_mean','MI_dir_L3_variance',
         'MI_dir_L1_weight','MI_dir_L1_mean','MI_dir_L1_variance','MI_dir_L0.1_weight','MI_dir_L0.1_mean','MI_dir_L0.1_variance',
         'MI_dir_L0.01_weight','MI_dir_L0.01_mean','MI_dir_L0.01_variance','H_L5_weight','H_L5_mean','H_L5_variance','H_L3_weight',
         'H_L3_mean','H_L3_variance','H_L1_weight','H_L1_mean','H_L1_variance','H_L0.1_weight','H_L0.1_mean','H_L0.1_variance',
         'H_L0.01_weight','H_L0.01_mean','H_L0.01_variance','HH_L5_weight','HH_L5_mean','HH_L5_std','HH_L5_magnitude','HH_L5_radius',
         'HH_L5_covariance','HH_L5_pcc','HH_L3_weight','HH_L3_mean','HH_L3_std','HH_L3_magnitude','HH_L3_radius','HH_L3_covariance',
         'HH_L3_pcc','HH_L1_weight','HH_L1_mean','HH_L1_std','HH_L1_magnitude','HH_L1_radius','HH_L1_covariance','HH_L1_pcc',
         'HH_L0.1_weight','HH_L0.1_mean','HH_L0.1_std','HH_L0.1_magnitude','HH_L0.1_radius','HH_L0.1_covariance','HH_L0.1_pcc',
         'HH_L0.01_weight','HH_L0.01_mean','HH_L0.01_std','HH_L0.01_magnitude','HH_L0.01_radius','HH_L0.01_covariance','HH_L0.01_pcc',
         'HH_jit_L5_weight','HH_jit_L5_mean','HH_jit_L5_variance','HH_jit_L3_weight','HH_jit_L3_mean','HH_jit_L3_variance',
         'HH_jit_L1_weight','HH_jit_L1_mean','HH_jit_L1_variance','HH_jit_L0.1_weight','HH_jit_L0.1_mean','HH_jit_L0.1_variance',
         'HH_jit_L0.01_weight','HH_jit_L0.01_mean','HH_jit_L0.01_variance','HpHp_L5_weight','HpHp_L5_mean','HpHp_L5_std','HpHp_L5_magnitude',
         'HpHp_L5_radius','HpHp_L5_covariance','HpHp_L5_pcc','HpHp_L3_weight','HpHp_L3_mean','HpHp_L3_std','HpHp_L3_magnitude',
         'HpHp_L3_radius','HpHp_L3_covariance','HpHp_L3_pcc','HpHp_L1_weight','HpHp_L1_mean','HpHp_L1_std','HpHp_L1_magnitude',
         'HpHp_L1_radius','HpHp_L1_covariance','HpHp_L1_pcc','HpHp_L0.1_weight','HpHp_L0.1_mean','HpHp_L0.1_std','HpHp_L0.1_magnitude',
         'HpHp_L0.1_radius','HpHp_L0.1_covariance','HpHp_L0.1_pcc','HpHp_L0.01_weight','HpHp_L0.01_mean','HpHp_L0.01_std','HpHp_L0.01_magnitude',
         'HpHp_L0.01_radius','HpHp_L0.01_covariance','HpHp_L0.01_pcc','Device','Label']

features = ['MI_dir_L5_weight','MI_dir_L5_mean','MI_dir_L5_variance','MI_dir_L3_weight','MI_dir_L3_mean','MI_dir_L3_variance',
         'MI_dir_L1_weight','MI_dir_L1_mean','MI_dir_L1_variance','MI_dir_L0.1_weight','MI_dir_L0.1_mean','MI_dir_L0.1_variance',
         'MI_dir_L0.01_weight','MI_dir_L0.01_mean','MI_dir_L0.01_variance','H_L5_weight','H_L5_mean','H_L5_variance','H_L3_weight',
         'H_L3_mean','H_L3_variance','H_L1_weight','H_L1_mean','H_L1_variance','H_L0.1_weight','H_L0.1_mean','H_L0.1_variance',
         'H_L0.01_weight','H_L0.01_mean','H_L0.01_variance','HH_L5_weight','HH_L5_mean','HH_L5_std','HH_L5_magnitude','HH_L5_radius',
         'HH_L5_covariance','HH_L5_pcc','HH_L3_weight','HH_L3_mean','HH_L3_std','HH_L3_magnitude','HH_L3_radius','HH_L3_covariance',
         'HH_L3_pcc','HH_L1_weight','HH_L1_mean','HH_L1_std','HH_L1_magnitude','HH_L1_radius','HH_L1_covariance','HH_L1_pcc',
         'HH_L0.1_weight','HH_L0.1_mean','HH_L0.1_std','HH_L0.1_magnitude','HH_L0.1_radius','HH_L0.1_covariance','HH_L0.1_pcc',
         'HH_L0.01_weight','HH_L0.01_mean','HH_L0.01_std','HH_L0.01_magnitude','HH_L0.01_radius','HH_L0.01_covariance','HH_L0.01_pcc',
         'HH_jit_L5_weight','HH_jit_L5_mean','HH_jit_L5_variance','HH_jit_L3_weight','HH_jit_L3_mean','HH_jit_L3_variance',
         'HH_jit_L1_weight','HH_jit_L1_mean','HH_jit_L1_variance','HH_jit_L0.1_weight','HH_jit_L0.1_mean','HH_jit_L0.1_variance',
         'HH_jit_L0.01_weight','HH_jit_L0.01_mean','HH_jit_L0.01_variance','HpHp_L5_weight','HpHp_L5_mean','HpHp_L5_std','HpHp_L5_magnitude',
         'HpHp_L5_radius','HpHp_L5_covariance','HpHp_L5_pcc','HpHp_L3_weight','HpHp_L3_mean','HpHp_L3_std','HpHp_L3_magnitude',
         'HpHp_L3_radius','HpHp_L3_covariance','HpHp_L3_pcc','HpHp_L1_weight','HpHp_L1_mean','HpHp_L1_std','HpHp_L1_magnitude',
         'HpHp_L1_radius','HpHp_L1_covariance','HpHp_L1_pcc','HpHp_L0.1_weight','HpHp_L0.1_mean','HpHp_L0.1_std','HpHp_L0.1_magnitude',
         'HpHp_L0.1_radius','HpHp_L0.1_covariance','HpHp_L0.1_pcc','HpHp_L0.01_weight','HpHp_L0.01_mean','HpHp_L0.01_std','HpHp_L0.01_magnitude',
         'HpHp_L0.01_radius','HpHp_L0.01_covariance','HpHp_L0.01_pcc']


def get_data_from_files(train, val, test):

    dftrain = pd.read_csv(train, names=names, header=None, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    dfval = pd.read_csv(val, names=names, header=None, sep=',', error_bad_lines=False, index_col=False,
                          dtype='unicode')
    dftest = pd.read_csv(test, names=names, header=None, sep=',', error_bad_lines=False, index_col=False,
                        dtype='unicode')

    lval = pd.DataFrame(data=dfval['Label'].values, columns=['Class'])
    ltest = pd.DataFrame(data=dftest['Label'].values, columns=['Class'])



    dftrain['Label'] = np.where(dftrain['Label'] == 'BENIGN', 0, 1)
    dfval['Label'] = np.where(dfval['Label'] == 'BENIGN', 0, 1)
    dftest['Label'] = np.where(dftest['Label'] == 'BENIGN', 0, 1)

    # ---- training set --- #
    dftrain = dftrain[dftrain['Label'] == 0]
    dftrain = dftrain.drop(['Label'], axis=1)
    dftrain = dftrain[features];

    x_train = dftrain.values

    # ---- validation set --- #
    x_val = dfval[features]
    y_val = dfval['Label']

    x_val = x_val.values;
    y_val = y_val.values;

    # ---- test set --- #
    x_test = dftest[features]
    y_test = dftest['Label']

    x_test = x_test.values;
    y_test = y_test.values;


    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_val, y_test, lval, ltest
