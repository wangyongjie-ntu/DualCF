#Filename:	gmsc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 12 Jul 2021 10:43:57 

import torch
import sys
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

def create_model(input_len):
    model = nn.Sequential(
            nn.Linear(input_len, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid(),
            )
    return model

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

def train(model, train_loader, optimizer, criterion, device):
    epoch_loss = 0
    prediction = []
    label = []
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        preds = torch.round(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(target)
        label.extend(target.tolist())
        prediction.extend(preds.reshape(-1).tolist())

    acc = accuracy_score(prediction, label)
    return epoch_loss / len(train_loader), acc


def test(model, test_loader, criterion, device):

    epoch_loss = 0
    prediction = []
    label = []
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.round(output)
            loss = criterion(output, target)
            epoch_loss += loss.item() * len(target)
            label.extend(target.tolist())
            prediction.extend(preds.tolist())
    
    prediction = np.array(prediction)
    label = np.array(label)
    acc = accuracy_score(prediction, label)

    return epoch_loss / len(test_loader), acc

if __name__ == "__main__":
    
    batch_size = 256
    epoches = 1000
    povit = int(sys.argv[1])

    x_cf = pd.read_csv("../train/adult-cf.csv")
    filename = '../../data/Adult/adult.csv'
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
    X_query = scaler.fit_transform(X_query)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)
    y_cf = y_cf[:, np.newaxis]

    y_query_ = np.load("../../data/Adult/y_query.npy")
    y_val_ = np.load("../../data/Adult/y_val.npy")
    y_test_ = np.load("../../data/Adult/y_test.npy")

    if povit != 0:
        x_query = x_query[0:povit]
        y_query_ = y_query_[0:povit]

    query_tensor = TensorDataset(torch.from_numpy(x_query), torch.from_numpy(y_query_))
    val_tensor = TensorDataset(torch.from_numpy(x_val_), torch.from_numpy(y_val_))
    test_tensor = TensorDataset(torch.from_numpy(x_test_), torch.from_numpy(y_test_))

    query_loader = DataLoader(query_tensor, batch_size = batch_size)
    val_loader = DataLoader(val_tensor, batch_size = batch_size)
    test_loader = DataLoader(test_tensor, batch_size = batch_size)

    model = create_model(x_query.shape[1])
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    criterion = nn.BCELoss()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):
        train_loss, train_acc = train(model, query_loader, optimizer, criterion, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Epoch: {epoch} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}')
    
    test_loss, test_acc = test(best_model, test_loader, criterion, device)

    print(f'Test Acc: {test_acc:.4f}')
