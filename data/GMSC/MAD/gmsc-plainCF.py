#Filename:	gmsc-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 13 Jul 2021 09:55:55 

import sys
sys.path.insert(0, '../../../cf/')

from nn import NNModel
from plainCF1 import PlainCF
from Dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import pandas as pd
import numpy as np


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

if __name__ == "__main__":
    
    thres = float(sys.argv[1])
    saved_name =  'gmsc-plaincf-mad.npy'
    data1, data2 = load_GMSC("../cs-training.csv")
    df = data1.iloc[0:int(len(data1) * 0.5)]
    df_query = data1.iloc[int(len(data1) * 0.5): int(len(data1) * 0.75), 1:]
    query = df_query.to_numpy().astype(np.float32)
    train_x = df.to_numpy()[:, 1:].astype(np.float32)
    scaler = StandardScaler()
    strain = scaler.fit_transform(train_x)
    squery = scaler.transform(query)

    clf = NNModel(model_path = '../gmsc_model.pt')
    cf = PlainCF(clf)

    gmsc_cf = np.zeros((df_query.shape[0], df_query.shape[1]+1))

    # determine the initialization
    y_pred = clf.predict_ndarray(strain)[0]
    idx0 = np.where(y_pred == 0)
    idx1 = np.where(y_pred == 1)
    init0 = strain[idx1].mean(0)
    init1 = strain[idx0].mean(0)

    # obtain mads
    mads = np.median(abs(train_x - np.median(train_x, 0)), 0)
    zero_index = np.where(mads == 0)
    mads[zero_index] = 1

    for i in range(len(query)):
        print(i)
        test_instance = squery[i:i+1]
        cur_pred = clf.predict_ndarray(test_instance)[0]
        if cur_pred == 0:
            init = init0
        else:
            init = init1

        lambda_ = 1
        init = init[np.newaxis, :]
        init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.01
        results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0004, max_iter = 2000, mads = mads)
        ylabel = clf.predict_ndarray(results)
        while ylabel[0] == cur_pred:
            init_final = init + np.random.rand(init.shape[0], init.shape[1]).astype(np.float32) * 0.01
            lambda_ += 1
            results = cf.generate_counterfactuals(test_instance, cur_pred, init_final, scaler, thres = thres, _lambda = lambda_, lr = 0.0004, max_iter = 2000, mads = mads)
            ylabel = clf.predict_ndarray(results)

        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        gmsc_cf[i, 1:] = plaincf_results
        gmsc_cf[i, 0] = ylabel[0]

    np.save(saved_name, gmsc_cf)

