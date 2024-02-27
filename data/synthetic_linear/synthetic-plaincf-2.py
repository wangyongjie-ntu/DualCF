#Filename:	synthetic-plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 28 Jul 2021 02:58:48 

import sys
sys.path.insert(0, '../../cf/')

from nn import NNModel
from plainCF import PlainCF
from sklearn.preprocessing import MinMaxScaler
from Dataset import Dataset
import numpy as np
import pandas as pd

if __name__ == "__main__":

    dtrain = pd.read_csv("synthetic_train.csv")
    dquery = np.load("91/synthetic-plaincf.npy")[:, 0:2]

    d = Dataset(dataframe = dtrain, continuous_features = ['X1', 'X2'], outcome_name = 'Y', scaler = MinMaxScaler())
    features_to_vary = ['X1', 'X2']
    clf = NNModel(model_path = './synthetic.pt')
    cf = PlainCF(d, clf)
    dquery = d.scaler.transform(dquery)
    
    synthetic_cf = np.zeros((len(dquery), 3))

    for i in range(len(dquery)):
        print(i)
        test_instance = dquery[i:i+1]
        results = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary, thres = 0.4, _lambda = 3, lr = 0.0004, max_iter = 5000)
        ylabel = clf.predict_ndarray(results)
        plaincf_results = d.denormalize_data(results)
        plaincf_results = np.round(plaincf_results, 2)
        synthetic_cf[i, 0:2] = plaincf_results
        synthetic_cf[i,2] = ylabel[0]

    np.save('91/synthetic-plaincf-2.npy', synthetic_cf)

