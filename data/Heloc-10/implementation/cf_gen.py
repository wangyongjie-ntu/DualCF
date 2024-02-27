#Filename:	cf_gen.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Okt 2021 11:25:05 

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from sklearn.datasets import load_boston
from alibi.explainers.cfproto import CounterfactualProto
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())

if __name__ == "__main__":
    df = pd.read_csv("./heloc_dataset_v1.csv")
    x_cols = list(df.columns.values)
    for col in x_cols:
        df[col][df[col].isin([-7, -8, -9])] = 0
    # Get the column names for the covariates and the dependent variable
    df = df[(df[x_cols].T != 0).any()]
    df['RiskPerformance'] = df['RiskPerformance'].map({'Good':1, 'Bad':0})
    df = df.astype(np.float32)
    columns = ['RiskPerformance', 'MSinceMostRecentInqexcl7days', 'ExternalRiskEstimate', 'NetFractionRevolvingBurden', 'NumSatisfactoryTrades', 'NumInqLast6M',
            'NumBank2NatlTradesWHighUtilization', 'AverageMInFile', 'NumRevolvingTradesWBalance', 'MaxDelq2PublicRecLast12M', 'PercentInstallTrades']

    df = df[columns]
    random.seed(0)
    a = list(range(len(df)))
    random.shuffle(a)
    length = len(a)
    print(a[0:10])

    train_x, train_y = df.iloc[a[0:int(len(a) * 0.5)], 1:].values, df.iloc[a[0:int(len(a) * 0.5)], 0].values
    query_x, query_y = df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 1:].values, df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 0].values
    test_x, test_y = df.iloc[a[int(len(a) * 0.75):], 1:].values, df.iloc[a[int(len(a) * 0.75):], 0].values

    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)
    stest_x = scaler.transform(test_x)
    otrain_y = to_categorical(train_y)
    oquery_y = to_categorical(query_y)
    otest_y = to_categorical(test_y)

    nn = load_model('nn_heloc_10.h5')
    print(nn.evaluate(squery_x, oquery_y))
    print(nn.evaluate(stest_x, otest_y))

    X = squery_x[1].reshape((1,) + squery_x[1].shape)
    shape = X.shape

    cf = CounterfactualProto(nn, shape, use_kdtree=True, theta=10., max_iterations=500,
                             feature_range=(strain_x.min(axis=0), strain_x.max(axis=0)),
                             c_init=1., c_steps=10)

    cf.fit(strain_x)

    query_cf = np.zeros_like(squery_x)
    query_cf_y = np.zeros(len(squery_x))

    for idx in range(len(squery_x)):
        print(idx)
        X = squery_x[idx].reshape((1,) + squery_x[idx].shape)
        explanation = cf.explain(X)
        query_cf[idx] = explanation.cf['X']
        query_cf_y[idx] = explanation.cf['class']

    tmp = np.concatenate((scaler.inverse_transform(query_cf), query_cf_y[:, np.newaxis]), axis = 1)
    np.save("heloc_query_cf.npy", tmp)

    query_ccf = np.zeros_like(squery_x)
    query_ccf_y = np.zeros(len(squery_x))

    for idx in range(len(query_cf)):
        print(idx)
        X = query_cf[idx].reshape((1,) + query_cf[idx].shape)
        explanation = cf.explain(X)
        query_ccf[idx] = explanation.cf['X']
        query_ccf_y[idx] = explanation.cf['class']

    tmp1 = np.concatenate((scaler.inverse_transform(query_ccf), query_ccf_y[:, np.newaxis]), axis = 1)
    np.save("heloc_query_2cf.npy", tmp1)
