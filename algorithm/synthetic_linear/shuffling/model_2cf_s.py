#Filename:	model_extraction.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 03 Agu 2021 07:45:12 

import numpy as np
import pandas as pd
import torch
import random
import copy
import sys
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    model.train()
    prediction = []
    label = []

    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.to(device), target.to(device)
        data, target = data.reshape(-1, 2), target.reshape(-1, 1)
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

    return epoch_loss / len(iterator.dataset), acc

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

    return epoch_loss / len(iterator.dataset), acc

if __name__ == "__main__":

    dquery = np.load("../../../data/synthetic_linear/synthetic_query_v1.npy").astype(np.float32)
    dtest = np.load("../../../data/synthetic_linear/synthetic_test_v1.npy").astype(np.float32)
    dcf = np.load("../../../data/synthetic_linear/64/synthetic-plaincf.npy").astype(np.float32)
    dccf = np.load("../../../data/synthetic_linear/64/synthetic-plaincf-2.npy").astype(np.float32)

    query_x, query_y = dquery[:, 0:2], dquery[:, 2:3]
    cf_x, cf_y = dcf[:, 0:2], dcf[:, 2:3]
    ccf_x, ccf_y = dccf[:, 0:2], dccf[:, 2:3]
    test_x, test_y = dtest[:, 0:2], dtest[:, 2:3]

    scaler = MinMaxScaler()
    query_x = scaler.fit_transform(query_x)
    test_x = scaler.transform(test_x)
    cf_x = scaler.transform(cf_x)
    ccf_x = scaler.transform(ccf_x)

    train_x = np.concatenate((cf_x, ccf_x), axis = 0)
    train_y = np.concatenate((cf_y, ccf_y), axis = 0)

    a = list(range(len(query_x)))
    length = int(sys.argv[1])
    if length == 0:
        length = len(a)
    
    print(length)
    acc_list = []

    for seed in range(100):
        print(seed)
        random.shuffle(a)
        idx = a[0:length]
        train_sx = train_x[idx]
        train_sy = train_y[idx]
        Query = TensorDataset(torch.from_numpy(train_sx), torch.from_numpy(train_sy))
        Test = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        query_loader = DataLoader(Query, batch_size  = 32)
        test_loader = DataLoader(Test, batch_size = 32)

        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

        optimizer = optim.Adam(model.parameters(), lr = 0.005)
        criterion = nn.BCELoss()
        device = torch.device("cpu")
        model = model.to(device)
        criterion = criterion.to(device)

        best_model = None
        best_acc = -float('inf')

        for epoch in range(200):
            train_loss, train_acc = train(model, query_loader, optimizer, criterion, device)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} |')
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Test acc: {test_acc:.4f}')
        acc_list.append(np.round(test_acc, 2))

    print(acc_list)
    print(np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3))

