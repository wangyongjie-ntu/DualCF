#Filename:	gmsc-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 13 Jul 2021 09:55:55 

import torch
import dice_ml
import pandas as pd
import numpy as np


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

    x_train, y_train = x_[0: int(len(x_) * 0.5)], y_[0: int(len(x_) * 0.5)]
    x_query, y_query = x_[int(len(x_) * 0.5): int(len(x_) * 0.65)], y_[int(len(y_) * 0.5):int(len(y_) * 0.65)]
    x_val, y_val = x_[int(len(x_) * 0.65):int(len(x_) * 0.8)], y_[int(len(y_) * 0.65):int(len(y_) * 0.8)]
    x_test, y_test = x_[int(len(x_) * 0.8): ], y_[int(len(y_) * 0.8):]

    return x_train, y_train, x_query, y_query, x_val, y_val, x_test, y_test


if __name__ == "__main__":

    data1, data2 = load_GMSC("../data/GMSC/cs-training.csv")
    x_train, y_train, x_query, y_query, x_val, y_val, x_test, y_test = train_test_split(data1)

    df = data1.iloc[0:int(len(data1) * 0.5)]
    df_query = data1.iloc[int(len(data1) * 0.5): int(len(data1) * 0.65), 1:]
    columns = data1.columns.tolist()

    outcome_name = columns[0]

    features = dict()
    for i in range(x_train.shape[1]):
        _min = x_train[:, i].min()
        _max = x_train[:, i].max()
        features[columns[i+1]] = [_min, _max]

    d = dice_ml.Data(dataframe = df, continuous_features = columns[1:], outcome_name = outcome_name)
    backend = 'PYT'
    model_path = "gmsc_model.pt"
    m = dice_ml.Model(model_path = model_path, backend = backend)
    exp = dice_ml.Dice(d, m)
    
    cf_df = None

    for i in range(len(df_query)):
        print(i)
        query_instance = df_query.iloc[i].to_dict()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs = 1, desired_class = 'opposite', verbose = False, posthoc_sparsity_algorithm=None, max_iter = 50000)
        if cf_df is None:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        else:
            cf_df = pd.concat((cf_df, dice_exp.cf_examples_list[0].final_cfs_df))
    
    cf_df.to_csv("gmsc-cf.csv", index = None)
