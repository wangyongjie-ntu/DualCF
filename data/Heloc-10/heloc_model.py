#Filename:	heloc_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 07 Okt 2021 01:52:46 
import pandas as pd
import numpy as np
import torch
import copy
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def create_model(input_len):
    model = nn.Sequential(
            nn.Linear(input_len, 20), #5, smaller
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
            )
    return model

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
    return epoch_loss / len(train_loader.dataset), acc


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

    return epoch_loss / len(test_loader.dataset), acc


if __name__ == "__main__":

    df = pd.read_csv("./heloc_dataset_v1.csv")
    x_cols = list(df.columns.values)
    for col in x_cols:
        df[col][df[col].isin([-7, -8, -9])] = 0
    # Get the column names for the covariates and the dependent variable
    df = df[(df[x_cols].T != 0).any()]
    df['RiskPerformance'] = df['RiskPerformance'].map({'Good':1, 'Bad':0})
    df = df.astype(np.float32)
    columns = ['RiskPerformance', 'MSinceMostRecentInqexcl7days', 'ExternalRiskEstimate', 'NetFractionRevolvingBurden', 'NumSatisfactoryTrades', 'NumInqLast6M', 
            'NumBank2NatlTradesWHighUtilization', 'AverageMInFile', 'NumRevolvingTradesWBalance', 'MaxDelq2PublicRecLast12M', 'PercentInstallTrades']
    
    df = df[columns]
    random.seed(0)
    a = list(range(len(df)))
    random.shuffle(a)
    length = len(a)
    print(a[0:10])

    train_x, train_y = df.iloc[a[0:int(len(a) * 0.5)], 1:].values, df.iloc[a[0:int(len(a) * 0.5)], 0].values
    query_x, query_y = df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 1:].values, df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)], 0].values
    test_x, test_y = df.iloc[a[int(len(a) * 0.75):], 1:].values, df.iloc[a[int(len(a) * 0.75):], 0].values

    scaler = StandardScaler()
    strain_x = scaler.fit_transform(train_x)
    squery_x = scaler.transform(query_x)
    stest_x = scaler.transform(test_x)

    batch_size = 128
    epoches = 500

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
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #optimizer = optim.SGD(model.parameters(), lr = 0.01)

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

    print(f'Test Acc: {test_acc:.4f}')
    torch.save(best_model, "heloc_model.pt")

    # use the prediction of f model as the label for reconstruction model
    best_model.eval()
    with torch.no_grad():
        x_query_t = torch.from_numpy(squery_x)
        x_test_t = torch.from_numpy(stest_x)
        y_query_t = torch.round(best_model(x_query_t)).numpy()
        y_test_t = torch.round(best_model(x_test_t)).numpy()

    
    tmp = np.concatenate((query_x, y_query_t), axis = 1)
    np.save("heloc_query.npy", tmp)
    tmp1 = np.concatenate((test_x, y_test_t), axis = 1)
    np.save("heloc_test.npy", tmp1)

