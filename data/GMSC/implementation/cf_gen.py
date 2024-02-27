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

def load_GMSC(filename):

    df = pd.read_csv(filename)
    df = df.drop("Unnamed: 0", axis=1) # drop id column
    df = df.loc[df["DebtRatio"] <= df["DebtRatio"].quantile(0.975)]
    df = df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] < 13)]
    df = df.loc[df["NumberOfTimes90DaysLate"] <= 17]
    dependents_mode = df["NumberOfDependents"].mode()[0] # impute with mode
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_mode)
    income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    mean = df["MonthlyIncome"].mean()
    std = df["MonthlyIncome"].std()
    df.loc[df["MonthlyIncome"].isnull()]["MonthlyIncome"] = np.random.normal(loc=mean, scale=std, size=len(df.loc[df["MonthlyIncome"].isnull()]))

    y = df['SeriousDlqin2yrs']
    idx1 = np.argwhere(y.values == 0).squeeze()
    idx2 = np.argwhere(y.values == 1).squeeze()
    idx3 = idx1[0:len(idx2)]
    idx4 = np.concatenate((idx2, idx3))
    np.random.seed(0)
    np.random.shuffle(idx4)
    idx5 = list(set(np.arange(len(df))).difference(idx4))

    data1 = df.iloc[idx4]
    data2 = df.iloc[idx5]

    return data1, data2

def train_test_split(data):

    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    x_ = x.to_numpy().astype(np.float32)
    y_ = y.to_numpy().astype(np.float32)
    y_ = y_[:, np.newaxis]

    x_train, y_train = x_[0: int(len(x_) * 0.5)], y_[0: int(len(y_) * 0.5)]
    x_query, y_query = x_[int(len(x_) * 0.5): int(len(x_) * 0.75)], y_[int(len(y_) * 0.5):int(len(y_) * 0.75)]
    x_test, y_test = x_[int(len(x_) * 0.75): ], y_[int(len(y_) * 0.75):]

    return x_train, y_train, x_query, y_query, x_test, y_test

if __name__ == "__main__":
    data1, data2 = load_GMSC('../cs-training.csv')
    train_x, train_y, query_x, query_y, test_x, test_y = train_test_split(data1)
    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)
    stest_x = scaler.transform(test_x)

    otrain_y = to_categorical(train_y)
    oquery_y = to_categorical(query_y)
    otest_y = to_categorical(test_y)

    nn = load_model('nn_GMSC.h5')
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
        try:
            query_cf[idx] = explanation.cf['X']
            query_cf_y[idx] = explanation.cf['class']
        except:
            query_cf_y[idx] = -1

    tmp = np.concatenate((query_cf_y[:, np.newaxis], scaler.inverse_transform(query_cf)), axis = 1)
    np.save("GMSC_query_cf.npy", tmp)

    query_ccf = np.zeros_like(squery_x)
    query_ccf_y = np.zeros(len(squery_x))

    for idx in range(len(query_cf)):
        print(idx)
        X = query_cf[idx].reshape((1,) + query_cf[idx].shape)
        explanation = cf.explain(X)
        try:
            query_ccf[idx] = explanation.cf['X']
            query_ccf_y[idx] = explanation.cf['class']
        except:
            query_ccf_y[idx] = -1

    tmp1 = np.concatenate((query_ccf_y[:, np.newaxis], scaler.inverse_transform(query_ccf)), axis = 1)
    np.save("GMSC_query_2cf.npy", tmp1)
