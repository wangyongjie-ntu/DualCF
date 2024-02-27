#Filename:	adult.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 07 Jul 2021 03:20:23 

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def create_model(input_len):

    model = nn.Sequential(
            nn.Linear(input_len, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
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

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    model.train()
    prediction = []
    label = []
    
    for batch_idx, (data, target) in enumerate(iterator):
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
        
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc, f1

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    model.eval()
    prediction = []
    label = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterator):
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            preds = torch.round(output)
            loss = criterion(output, target)
            epoch_loss += loss.item() * len(target)
            label.extend(target.tolist())
            prediction.extend(preds.reshape(-1).tolist())
            
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc, f1

if __name__ == "__main__":

    epoches = 100
    saved_name = "../weights/adult.pth"
    filename = "../data/Adult/adult.csv"
    batch_size = 128

    income_df = load_adult_income(filename)
    X_train, y_train, X_query, y_query, X_val, y_val, X_test, y_test = split_data(income_df)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_query = scaler.transform(X_query)

    Train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    Val = TensorDataset(torch.from_numpy(X_val),  torch.from_numpy(y_val))
    Test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(Train, batch_size = batch_size)
    val_loader = DataLoader(Val, batch_size = batch_size)
    test_loader = DataLoader(Test, batch_size = batch_size)

    model = create_model(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    criterion = nn.BCELoss()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    best_f1 = -float('inf')
    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc, val_f1 = evaluate(model, test_loader, criterion, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model)
            best_acc = valid_acc

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |  Val. F1: {val_f1:.4f}')

    print("Best Val. F1: {:.4f}, Best Val. Accuarcy: {:.4f}".format(best_f1, best_acc))

    torch.save(best_model, "adult.pt")

    X_query = torch.from_numpy(X_query)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)

    best_model.eval()
    with torch.no_grad():
        y_query_p = torch.round(model(X_query)).numpy()
        y_val_p = torch.round(model(X_val)).numpy()
        y_test_p = torch.round(model(X_test)).numpy()

    np.save("../data/Adult/y_query.npy", y_query_p)
    np.save("../data/Adult/y_val.npy", y_val_p)
    np.save("../data/Adult/y_test.npy", y_test_p)
