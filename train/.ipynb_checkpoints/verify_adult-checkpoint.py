#Filename:	verify_adult.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 18 Jul 2021 12:54:54 

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

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

def split_data(income_df):

    y = income_df.iloc[:, -1]
    X = income_df.iloc[:, 0:-1]

    # find all categorical features
    categorical = []
    for col in income_df.columns:
        if income_df[col].dtype == object and col != "income":
            categorical.append(col)

    X = pd.get_dummies(columns = categorical, data = X, prefix = categorical, prefix_sep="_")
    X_, y_ = X.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)
    y_ = y_[:, np.newaxis]

    x_train, y_train = X_[0: int(len(X_) * 0.5)], y_[0: int(len(y_) * 0.5)]
    x_query, y_query = X_[int(len(X_) * 0.5): int(len(X_) * 0.65)], y_[int(len(y_) * 0.5):int(len(y_) * 0.65)]
    x_val, y_val = X_[int(len(X_) * 0.65):int(len(X_) * 0.8)], y_[int(len(y_) * 0.65):int(len(y_) * 0.8)]
    x_test, y_test = X_[int(len(X_) * 0.8): ], y_[int(len(y_) * 0.8):]

    return x_train, y_train, x_query, y_query, x_val, y_val, x_test, y_test


if __name__ == "__main__":

    x_cf = pd.read_csv("./adult-cf.csv")
    filename = '../data/Adult/adult.csv'
    # find all categorical features
    categorical = []
    for col in x_cf.columns:
        if x_cf[col].dtype == object and col != "income":
            categorical.append(col)

    x_cf = pd.get_dummies(columns = categorical, data = x_cf, prefix = categorical, prefix_sep="_")
    x_cf, y_cf = x_cf.iloc[:,0:-1].to_numpy().astype(np.float32), x_cf.iloc[:, -1].to_numpy().astype(np.float32)
    income_df = load_adult_income(filename)
    X_train, y_train, X_query, y_query, X_val, y_val, X_test, y_test = split_data(income_df)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    x_cf = scaler.transform(x_cf)
    y_cf = y_cf[:, np.newaxis]

    model = torch.load('./adult.pt')
    model.eval()

    with torch.no_grad():
        x_cf = torch.from_numpy(x_cf)

    pred_y = torch.round(model(x_cf))

    _sum = pred_y.detach().numpy() + y_query
    print((_sum == 1).sum() / len(y_query))
