#Filename:	gmsc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 12 Jul 2021 10:43:57 

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

def create_model(input_len):
    model = nn.Sequential(
            nn.Linear(input_len, 20), #5, smaller
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(20, 10),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid(),
            )
    return model

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

def train_test_split(data):

    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    x_ = x.to_numpy().astype(np.float32)
    y_ = y.to_numpy().astype(np.float32)
    y_ = y_[:, np.newaxis]

    x_train, y_train = x_[0: int(len(x_) * 0.5)], y_[0: int(len(y_) * 0.5)]
    x_query, y_query = x_[int(len(x_) * 0.5): int(len(x_) * 0.75)], y_[int(len(y_) * 0.5):int(len(y_) * 0.75)]
    x_test, y_test = x_[int(len(x_) * 0.75): ], y_[int(len(y_) * 0.75):]

    return x_train, y_train, x_query, y_query, x_test, y_test

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
    
    batch_size = 128
    epoches = 500

    data1, data2 = load_GMSC('cs-training.csv')
    train_x, train_y, query_x, query_y, test_x, test_y = train_test_split(data1)
    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)
    stest_x = scaler.transform(test_x)
    '''
    train_tensor = TensorDataset(torch.from_numpy(strain_x), torch.from_numpy(train_y))
    val_tensor = TensorDataset(torch.from_numpy(squery_x), torch.from_numpy(query_y))
    test_tensor = TensorDataset(torch.from_numpy(stest_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_tensor, batch_size = batch_size)
    val_loader = DataLoader(val_tensor, batch_size = batch_size)
    test_loader = DataLoader(test_tensor, batch_size = batch_size)

    model = create_model(train_x.shape[1])
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    criterion = nn.BCELoss()
    model = model.to(device)
    criterion = criterion.to(device)
    #optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    optimizer = optim.SGD(model.parameters(), lr = 0.005)

    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Epoch: {epoch} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}')
    
    test_loss, test_acc = test(best_model, test_loader, criterion, device)
    
    best_model = torch.load("gmsc_model.pt")
    print(f'Test Acc: {test_acc:.4f}')
    torch.save(best_model, "gmsc_model.pt")
    '''

    best_model = torch.load("gmsc_model.pt")
    # use the prediction of f model as the label for reconstruction model

    best_model.eval()
    with torch.no_grad():
        x_query_t = torch.from_numpy(squery_x)
        x_test_t = torch.from_numpy(stest_x)
        y_query_t = torch.round(best_model(x_query_t)).numpy()
        y_test_t = torch.round(best_model(x_test_t)).numpy()

    
    tmp = np.concatenate((y_query_t, query_x), axis = 1)
    np.save("gmsc_query.npy", tmp)
    tmp1 = np.concatenate((y_test_t, test_x), axis = 1)
    np.save("gmsc_test.npy", tmp1)
