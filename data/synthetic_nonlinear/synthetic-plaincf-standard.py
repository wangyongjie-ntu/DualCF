#Filename:	synthetic-plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 28 Jul 2021 02:58:48 

import sys
sys.path.insert(0, '../../cf/')

from nn import NNModel
from plainCF_standard import PlainCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Dataset import Dataset
import numpy as np
import pandas as pd

if __name__ == "__main__":

    dquery = pd.read_csv("synthetic_query.csv")
    dtrain = pd.read_csv("synthetic_train.csv")

    train_x, train_y = dtrain.iloc[:, 0:2].astype(np.float32), dtrain.iloc[:, -1].astype(np.int)
    query_x, query_y = dquery.iloc[:, 0:2].astype(np.float32), dquery.iloc[:, -1].astype(np.int)

    clf = NNModel(model_path = './synthetic.pt')
    cf = PlainCF(clf)
    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)

    synthetic_cf = np.zeros((len(dquery), 3))
    
    maxx = strain_x.max(0)
    minx = strain_x.min(0)
    y_pred = clf.predict_ndarray(strain_x)[0]
    idx0 = np.where(y_pred == 0)
    idx1 = np.where(y_pred == 1)
    init0 = strain_x[idx1].mean(0)
    init1 = strain_x[idx0].mean(0)

    for i in range(len(dquery)):
        print(i)
        cur_pred = query_y[i]
        if cur_pred == 0:
            init = init0
        else:
            init = init1

        lambda_ = 5
        init = init[np.newaxis, :]
        test_instance = squery_x[i:i+1]
        results = cf.generate_counterfactuals(test_instance, cur_pred, init, scaler, maxx, minx, _lambda = lambda_, lr = 0.0001, max_iter = 2000, mads = None)
        ylabel = clf.predict_ndarray(results)
        while ylabel[0] == cur_pred:
            lambda_ += 1
            results = cf.generate_counterfactuals(test_instance, cur_pred, init, scaler, maxx, minx, _lambda = lambda_, lr = 0.0001, max_iter = 2000, mads = None)
            ylabel = clf.predict_ndarray(results)

        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        synthetic_cf[i, 0:2] = plaincf_results
        synthetic_cf[i,2] = ylabel[0]

    np.save('synthetic-plaincf-standard.npy', synthetic_cf)

