#Filename:	gmsc-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 13 Jul 2021 09:55:55 

import sys
sys.path.insert(0, '../../../cf/')

from nn import NNModel
from plainCF2 import PlainCF

import torch
import copy
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":

    saved_name = 'synthetic-plaincf-l1.npy'

    dquery = pd.read_csv("../synthetic_query.csv")
    dtrain = pd.read_csv("../synthetic_train.csv")

    train_x, train_y = dtrain.iloc[:, 0:2].values, dtrain.iloc[:, 2].values
    query_x, query_y = dquery.iloc[:, 0:2].values, dquery.iloc[:, 2].values

    scaler = MinMaxScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)

    clf = NNModel(model_path = '../synthetic.pt')
    cf = PlainCF(clf)

    synthetic_cf = np.zeros((query_x.shape[0], 3))

    for i in range(len(squery_x)):
        print(i)
        test_instance = squery_x[i:i+1]
        cur_pred = query_y[i]
        if cur_pred == 0:
            init = np.array([0.9, 0.1]).astype(np.float32)
        else:
            init = np.array([0.1, 0.9]).astype(np.float32)

        init = init[np.newaxis, :]
        init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.01
        results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = 0.1, _lambda = 3, lr = 0.0004, max_iter = 2000, mads = None)
        ylabel = clf.predict_ndarray(results)
        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        synthetic_cf[i, 0:-1] = plaincf_results
        synthetic_cf[i, -1] = ylabel[0]
    
    np.save(saved_name, synthetic_cf)
