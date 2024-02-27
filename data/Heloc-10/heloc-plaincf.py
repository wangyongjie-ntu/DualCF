#Filename:	gmsc-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 13 Jul 2021 09:55:55 

import sys
sys.path.insert(0, '../../cf/')

from nn import NNModel
from plainCF1 import PlainCF

import torch
import copy
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":

    thres = float(sys.argv[1])
    saved_name = str(sys.argv[2])
    saved_name = saved_name + '/heloc-plaincf-l2.npy'

    df = pd.read_csv("./heloc_dataset_v1.csv")
    query = np.load("./heloc_query.npy")
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

    train_x, train_y = df.iloc[a[0:int(len(a) * 0.5)], 1:].values, df.iloc[a[0:int(len(a) * 0.5)], 0].values
    #query_x, query_y = df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 1:].values, df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 0].values
    query_x, query_y = query[:, 0:-1], query[:, -1].astype(np.int)

    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)

    clf = NNModel(model_path = './heloc_model.pt')
    cf = PlainCF(clf)

    heloc_cf = np.zeros((query_x.shape[0], query_x.shape[1]+1))

    # determine the initialization
    y_pred = clf.predict_ndarray(strain_x)[0]
    idx0 = np.where(y_pred == 0)
    idx1 = np.where(y_pred == 1)
    init0 = strain_x[idx1].mean(0)
    init1 = strain_x[idx0].mean(0)

    for i in range(len(squery_x)):
        print(i)
        test_instance = squery_x[i:i+1]
        cur_pred = query_y[i]
        if cur_pred == 0:
            init = init0
        else:
            init = init1

        lambda_ = 1
        init = init[np.newaxis, :]
        init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.1
        results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0004, max_iter = 2000, mads = None)
        ylabel = clf.predict_ndarray(results)
        while ylabel[0] == cur_pred:
            init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.2
            lambda_ += 1
            results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0004, max_iter = 2000, mads = None)
            ylabel = clf.predict_ndarray(results)

        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        heloc_cf[i, 0:-1] = plaincf_results
        heloc_cf[i, -1] = ylabel[0]
    
    np.save(saved_name, heloc_cf)

