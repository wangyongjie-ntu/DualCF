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
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":
    
    thres = float(sys.argv[1])
    name = str(sys.argv[2])
    saved_name = name + '/boston-plaincf-l2.npy'
    boston = load_boston()
    data = boston.data
    target = boston.target
    feature_names = boston.feature_names

    y = np.zeros((target.shape[0],))
    y[np.where(target > np.median(target))[0]] = 1

    data = np.delete(data, 3, 1)
    feature_names = np.delete(feature_names, 3)
    

    data = data.astype(np.float32)
    y = y.astype(np.float32)
    random.seed(0)
    a = list(range(len(data)))
    random.shuffle(a)
    length = len(a)

    train_x, train_y = data[a[0:int(0.5*length)]], y[a[0:int(0.5*length)]]
    query_x, query_y = data[a[int(0.5*length):int(0.75*length)]], y[a[int(0.5*length):int(0.75*length)]]
    test_x, test_y = data[a[int(0.75*length):]], y[a[int(0.75*length):]]

    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)
    stest_x = scaler.transform(test_x)

    clf = NNModel(model_path = './boston_model.pt')
    cf = PlainCF(clf)

    boston_cf = np.zeros((query_x.shape[0], query_x.shape[1]+1))

    # determine the initialization
    y_pred = clf.predict_ndarray(strain_x)[0]
    idx0 = np.where(y_pred == 0)
    idx1 = np.where(y_pred == 1)
    init0 = strain_x[idx1].mean(0)
    init1 = strain_x[idx0].mean(0)

    # obtain mads
    #mads = np.median(abs(train_x - np.median(train_x, 0)), 0)
    #zero_index = np.where(mads == 0)
    #mads[zero_index] = 1

    for i in range(len(squery_x)):
        print(i)
        test_instance = squery_x[i:i+1]
        cur_pred = clf.predict_ndarray(test_instance)[0]
        if cur_pred == 0:
            init = init0
        else:
            init = init1
        
        lambda_ = 1
        init = init[np.newaxis, :]
        init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.1
        results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0005, max_iter = 2000, mads = None)
        ylabel = clf.predict_ndarray(results)
        while ylabel[0] == cur_pred:
            init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.1
            lambda_ += 1
            results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0005, max_iter = 2000, mads = None)
            ylabel = clf.predict_ndarray(results)

        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        boston_cf[i, 0:-1] = plaincf_results
        boston_cf[i, -1] = ylabel[0]

    np.save(saved_name, boston_cf)

