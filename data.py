from __future__ import unicode_literals
from sklearn.preprocessing import MaxAbsScaler, Normalizer, RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer

import pandas as pd
import numpy as np

np.random.seed(101)

names = ['StartTime','LastTime','SrcAddr','DstAddr','Mean','Sport','Dport','SrcPkts','DstPkts','TotPkts','DstBytes',
         'SrcBytes','TotBytes','SrcLoad','DstLoad','Load','SrcRate','DstRate','Rate','SrcLoss','DstLoss','Loss','pLoss',
         'SrcJitter','DstJitter','SIntPkt','DIntPkt','Proto','Dur','TcpRtt','IdleTime','Sum','Min','Max','sDSb','sTtl',
         'dTtl','sIpId','dIpId','SAppBytes','DAppBytes','TotAppByte','SynAck','RunTime','sTos','SrcJitAct','DstJitAct','Label']

features = ['Mean','Sport','Dport','SrcPkts','DstPkts','TotPkts','DstBytes',
         'SrcBytes','TotBytes','SrcLoad','DstLoad','Load','SrcRate','DstRate','Rate','SrcLoss','DstLoss','Loss','pLoss',
         'SrcJitter','DstJitter','SIntPkt','DIntPkt','Proto','Dur','TcpRtt','IdleTime','Sum','Min','Max','sDSb','sTtl',
         'dTtl','SAppBytes','DAppBytes','TotAppByte','SynAck','RunTime','sTos','SrcJitAct','DstJitAct']


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


    scaler = QuantileTransformer(n_quantiles=4)
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_val, y_test, lval, ltest

