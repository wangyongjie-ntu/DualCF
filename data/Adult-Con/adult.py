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
            nn.Dropout(0.2),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, 1),
            nn.Sigmoid()
            )

    return model

def load_adult_income(filename):

    income_df = pd.read_csv(filename)
    income_df.replace("?", np.nan, inplace= True)
    income_df.drop(["fnlwgt"], axis = 1, inplace = True)
    income_df.at[income_df[income_df['income'] == '>50K'].index, 'income'] = 1
    income_df.at[income_df[income_df['income'] == '<=50K'].index, 'income'] = 0

    for col in income_df.columns:
        if income_df[col].dtype == np.float64 or income_df[col].dtype == np.int64:
            income_df[col].fillna(income_df[col].mean(), inplace = True)
        elif income_df[col].dtype  == object:
            income_df[col].fillna(income_df[col].mode()[0], inplace = True)
        else:
            continue

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

    income_df['native-country'] = income_df['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
    income_df['education'] = income_df['education'].map({'dropout':1, 'HighGrad':2, 'CommunityCollege':3, 'Bachelors':4, 'Masters':5, "Doctorate":6})
    income_df['education'] = income_df['education'].astype(np.int64)
    income_df['native-country'] = income_df['native-country'].astype(np.int64)
    income_df['income'] = income_df['income'].astype(np.int64)

    con_columns = []
    for col in income_df.columns:
        if income_df[col].dtype == np.float64 or income_df[col].dtype == np.int64:
            con_columns.append(col)

    income_df_con = income_df[con_columns]

    return income_df_con

def split_data(income_df):

    y = income_df.iloc[:, -1]
    X = income_df.iloc[:, 0:-1]

    X_, y_ = X.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)
    y_ = y_[:, np.newaxis]
    x_train, y_train = X_[0: int(len(X_) * 0.8)], y_[0: int(len(y_) * 0.8)]
    x_query, y_query = X_[int(len(X_) * 0.8): int(len(X_) * 0.9)], y_[int(len(y_) * 0.8):int(len(y_) * 0.9)]
    x_test, y_test = X_[int(len(X_) * 0.9): ], y_[int(len(y_) * 0.9):]

    return x_train, y_train, x_query, y_query, x_test, y_test

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

    epoches = 200
    saved_name = "adult.pth"
    filename = "adult.csv"
    batch_size = 64

    income_df = load_adult_income(filename)
    X_train, y_train, X_query, y_query, X_test, y_test = split_data(income_df)
    scaler = MinMaxScaler()
    sx_train = scaler.fit_transform(X_train)
    sx_test = scaler.transform(X_test)
    sx_query = scaler.transform(X_query)

    Train = TensorDataset(torch.from_numpy(sx_train), torch.from_numpy(y_train))
    Query = TensorDataset(torch.from_numpy(sx_query),  torch.from_numpy(y_query))
    Test = TensorDataset(torch.from_numpy(sx_test), torch.from_numpy(y_test))

    train_loader = DataLoader(Train, batch_size = batch_size)
    query_loader = DataLoader(Query, batch_size = batch_size)
    test_loader = DataLoader(Test, batch_size = batch_size)

    model = create_model(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #optimizer = optim.SGD(model.parameters(), lr = 0.1)
    criterion = nn.BCELoss()
    device = torch.device("cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    best_f1 = -float('inf')
    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc, val_f1 = evaluate(model, query_loader, criterion, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model)
            best_acc = valid_acc
       
        if epoches % 50 == 0:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
            print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |  Val. F1: {val_f1:.4f}')

    print("Best Val. F1: {:.4f}, Best Val. Accuarcy: {:.4f}".format(best_f1, best_acc))

    test_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion, device)
    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    torch.save(best_model, "adult.pt")

    sx_query = torch.from_numpy(sx_query)
    sx_test = torch.from_numpy(sx_test)

    best_model.eval()
    with torch.no_grad():
        y_query_p = torch.round(best_model(sx_query)).numpy()
        y_test_p = torch.round(best_model(sx_test)).numpy()
    
    tmp = np.concatenate((X_query, y_query_p), axis = 1)
    np.save("adult_query.npy", tmp)
    tmp1 = np.concatenate((X_test, y_test_p), axis = 1)
    np.save("adult_test.npy", tmp1)
