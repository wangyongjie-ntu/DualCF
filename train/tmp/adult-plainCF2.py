#Filename:	gmsc-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 13 Jul 2021 09:55:55 

import sys
sys.path.insert(0, '../cf/')

from nn import NNModel
from plainCF1 import PlainCF
from Dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np

def load_adult_income(filename):

    income_df = pd.read_csv(filename)
    income_df.replace("?", np.nan, inplace= True)

    for col in income_df.columns:
        if income_df[col].dtype == np.float64:
            income_df[col].fillna(income_df[col].mean()[0], inplace = True)
        elif income_df[col].dtype  == object:
            income_df[col].fillna(income_df[col].mode()[0], inplace = True)
        else:
            continue
    
    income_df.drop(["fnlwgt"], axis = 1, inplace = True)
    income_df.at[income_df[income_df['income'] == '>50K'].index, 'income'] = 1
    income_df.at[income_df[income_df['income'] == '<=50K'].index, 'income'] = 0

    income_df['education'].replace('Preschool', 'dropout',inplace=True)
    income_df['education'].replace('10th', 'dropout',inplace=True)
    income_df['education'].replace('11th', 'dropout',inplace=True)
    income_df['education'].replace('12th', 'dropout',inplace=True)
    income_df['education'].replace('1st-4th', 'dropout',inplace=True)
    income_df['education'].replace('5th-6th', 'dropout',inplace=True)
    income_df['education'].replace('7th-8th', 'dropout',inplace=True)
    income_df['education'].replace('9th', 'dropout',inplace=True)
    income_df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
    income_df['education'].replace('HS-grad', 'HighGrad',inplace=True)
    income_df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Bachelors', 'Bachelors',inplace=True)
    income_df['education'].replace('Masters', 'Masters',inplace=True)
    income_df['education'].replace('Prof-school', 'Doctorate',inplace=True)
    income_df['education'].replace('Doctorate', 'Doctorate',inplace=True)

    income_df = income_df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    income_df = income_df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    income_df = income_df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'?': 'Other/Unknown'}})


    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }

    income_df['occupation'] = income_df['occupation'].map(occupation_map)

    income_df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

    income_df['native-country'] = income_df['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
    income_df['education'] = income_df['education'].map({'dropout':1, 'HighGrad':2, 'CommunityCollege':3, 'Bachelors':4, 'Masters':5, "Doctorate":6})

    return income_df

if __name__ == "__main__":

    income_df = load_adult_income("../data/Adult/adult.csv")
    df_query = pd.read_csv("./adult-plaincf.csv")
    columns = income_df.columns.tolist()
    continuous_features = []
    categorical = []

    for col in income_df.columns:
        if income_df[col].dtype == object and col != "income":
            categorical.append(col)
        if income_df[col].dtype != object and col != 'income':
            continuous_features.append(col)

    maps_encode = list()

    for i in range(len(categorical)):
        maps_encode.append(income_df[categorical[i]].unique().tolist())

    y = income_df.iloc[:, -1]
    X = income_df.iloc[:, 0:-1]
    X = pd.get_dummies(columns = categorical, data = X, prefix = categorical, prefix_sep="_")
    X_, y_ = X.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)
    
    x_query = df_query.iloc[:, 0:-1]
    y_query = df_query.iloc[:, -1]
    x_query = pd.get_dummies(columns = categorical, data = x_query, prefix = categorical, prefix_sep = "_").to_numpy().astype(np.float32)

    X_train, Y_train = X_[0: int(len(X_) * 0.5)], y_[0: int(len(y_) * 0.5)]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_query = scaler.transform(x_query)

    clf = NNModel(model_path = '../weights/adult.pth')
    cf = PlainCF(clf)

    adult_cf = np.zeros((len(x_query), income_df.shape[1]))
    adult_cf = pd.DataFrame(adult_cf, columns = continuous_features + categorical + ['income'])

    # determine the initialization
    y_pred = clf.predict_ndarray(x_train)[0]
    idx0 = np.where(y_pred == 0)
    idx1 = np.where(y_pred == 1)
    init0 = x_train[idx1].mean(0)
    init1 = x_train[idx0].mean(0)

    # obtain mads
    mads = np.median(abs(X_train - np.median(X_train, 0)), 0)
    zero_index = np.where(mads == 0)
    mads[zero_index] = 1

    for i in range(len(x_query)):
        print(i)
        test_instance = x_query[i:i+1]
        cur_pred = int(y_query[i])

        if cur_pred == 0:
            init = init1
        else:
            init = init0

        init = init[np.newaxis, :]
        results = cf.generate_counterfactuals(test_instance, cur_pred, init, scaler, _lambda = 1, lr = 0.0004, max_iter = 5000, mads = mads)
        ylabel = clf.predict_ndarray(results)
        plaincf_results = scaler.inverse_transform(results)
        plaincf_results = np.round(plaincf_results, 2)
        # copy the continuous features
        adult_cf.iloc[i, 0:len(continuous_features)] = plaincf_results[0, 0:len(continuous_features)]
        # copy the categorical features
        flag = len(continuous_features)
        idx = len(continuous_features)
        for j in range(len(categorical)):
            tmp = plaincf_results[0, flag:flag + len(maps_encode[j])]
            max_loc = tmp.argmax()
            adult_cf.iloc[i, idx] = maps_encode[j][max_loc]
            flag += len(maps_encode[j])
            idx += 1

        adult_cf.iloc[i, -1] = ylabel[0]

    adult_cf = adult_cf[columns]
    adult_cf.to_csv("adult-plaincf2.csv", index = None)

