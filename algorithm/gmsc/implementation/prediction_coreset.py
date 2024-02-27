#Filename:	prediction_coreset.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 16 Des 2021 02:39:40 

import numpy as np
import pandas as pd
import torch
import random
import copy
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, "../../../util/")
import coreset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

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

    dquery = np.load("../../../data/GMSC/implementation/gmsc_query.npy").astype(np.float32)
    dtest = np.load("../../../data/GMSC/implementation/gmsc_test.npy").astype(np.float32)

    query_x, query_y = dquery[:, 1:], dquery[:, 0:1]
    test_x, test_y = dtest[:, 1:], dtest[:, 0:1]

    scaler = StandardScaler()
    query_x = scaler.fit_transform(query_x)
    test_x = scaler.transform(test_x)
    input_len = query_x.shape[1]
    length = int(sys.argv[1])
    if length == 0:
        length = len(a)
    print(length)
    acc_list = []

    for seed in range(100):
        #random.seed(seed)
        idx = coreset.find_core_set(query_x, length)
        query_sx, query_sy = query_x[idx], query_y[idx]

        Query = TensorDataset(torch.from_numpy(query_sx), torch.from_numpy(query_sy))
        Test = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        query_loader = DataLoader(Query, batch_size  = 32)
        test_loader = DataLoader(Test, batch_size = 32)

        model = nn.Sequential(
                nn.Linear(input_len, 10),
                nn.ReLU(),
                nn.Dropout(0.2),
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

        for epoch in range(500):
            train_loss, train_acc = train(model, query_loader, optimizer, criterion, device)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Runs {seed}. Test acc: {test_acc:.4f}')
        acc_list.append(np.round(test_acc, 2))

    print(acc_list)
    print(np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3))

