#Filename:	verify_gmsc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 16 Jul 2021 07:39:12 

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def load_GMSC(filename):

    df = pd.read_csv(filename)
    #df.SeriousDlqin2yrs = 1 - df.SeriousDlqin2yrs # for simply computing the CF
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
    x_query, y_query = x_[int(len(x_) * 0.5): int(len(x_) * 0.65)], y_[int(len(y_) * 0.5):int(len(y_) * 0.65)]
    x_val, y_val = x_[int(len(x_) * 0.65):int(len(x_) * 0.8)], y_[int(len(y_) * 0.65):int(len(y_) * 0.8)]
    x_test, y_test = x_[int(len(x_) * 0.8): ], y_[int(len(y_) * 0.8):]

    return x_train, y_train, x_query, y_query, x_val, y_val, x_test, y_test


if __name__ == "__main__":

    data1, data2 = load_GMSC('../data/GMSC/cs-training.csv')
    x_train, y_train, x_query, y_query, x_val, y_val, x_test, y_test = train_test_split(data1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)

    cf = np.load("gmsc-plaincf.npy").astype(np.float32)
    x_cf = cf[:, 1:]
    x_cf = scaler.transform(x_cf)
    x_query = scaler.transform(x_query)

    model = torch.load("../weights/gmsc_model.pt")
    model.eval()
    with torch.no_grad():
        x_cf = torch.from_numpy(x_cf)
        x_query = torch.from_numpy(x_query)

    pred_y = torch.round(model(x_cf))
    pred_y1 = torch.round(model(x_query))

    _sum = pred_y.detach().numpy() + pred_y1.detach().numpy()

    print((_sum == 1).sum() / len(y_query))

