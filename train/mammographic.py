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


def load_mammographic(filename):



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

def train_adult_income(filename, epoches, save_name):

    income_df = load_adult_income(filename)
    X_train, X_test, y_train, y_test, columns = split_data(income_df)
    train_loader, test_loader, scaler = data_loader_torch(X_train, X_test, y_train, y_test)
    model = create_model(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    best_f1 = -float('inf')
    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, val_f1 = evaluate(model, test_loader, criterion)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model)
            best_acc = valid_acc

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |  Val. F1: {val_f1:.4f}')

    print("Best Val. F1: {:.4f}, Best Val. Accuarcy: {:.4f}".format(best_f1, best_acc))

    torch.save(best_model, save_name)

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

    X_train, X_test, y_train, y_test =  train_test_split(X_, y_, test_size = 0.2, shuffle = False, random_state = 0)

    return X_train, X_test, y_train, y_test, X.columns

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    model.train()
    prediction = []
    label = []
    
    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.cuda(), target.cuda()
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

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    model.eval()
    prediction = []
    label = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterator):
            
            data, target = data.cuda(), target.cuda()
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

def data_loader_torch(X_train, X_test, y_train, y_test, batch_size = 128):

    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    Test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(Train, batch_size = batch_size)
    test_loader = DataLoader(Test, batch_size = batch_size)

    return train_loader, test_loader, scaler

if __name__ == "__main__":

    epoch = 100
    saved_name = "../weights/adult.pth"
    filename = "../data/Adult/adult.csv"
    train_adult_income(filename, epoch, saved_name)

