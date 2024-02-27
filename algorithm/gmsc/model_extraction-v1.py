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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

def create_model(input_len):
    model = nn.Sequential(
            nn.Linear(input_len, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid(),
            )
    return model

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

    batch_size = 128
    epoches = 500

    x_query, y_query = np.load("../../data/GMSC_V1/X_query.npy"), np.load("../../data/GMSC_V1/y_query.npy")
    x_test, y_test = np.load("../../data/GMSC_V1/X_test.npy"), np.load("../../data/GMSC_V1/y_test.npy")
    cf = np.load("../../data/GMSC_V1/gmsc-plaincf.npy").astype(np.float32)
    dcf = np.load("../../data/GMSC_V1/gmsc-plaincf2.npy").astype(np.float32)
    x_cf, y_cf = cf[:, 1:], cf[:, 0]
    dx_cf, dy_cf = dcf[:, 1:], dcf[:, 0]
    y_cf = y_cf[:, np.newaxis]
    dy_cf = dy_cf[:, np.newaxis]

    scaler = MinMaxScaler()
    x_query = scaler.fit_transform(x_query)
    x_test_ = scaler.transform(x_test)
    x_cf = scaler.transform(x_cf)
    dx_cf = scaler.transform(dx_cf)

    a = list(range(len(x_query)))
    length = int(sys.argv[1])
    if length == 0:
        length = len(a)

    acc_list = []

    for seed in range(30):
        print("seed used :{}".format(seed))
        #random.seed(seed)
        random.shuffle(a)
        idx = a[0:length]
        cf_sx, cf_sy = x_cf[idx], y_cf[idx]
        cf_sx1, cf_sy1 = dx_cf[idx], dy_cf[idx]

        train_x, train_y = np.concatenate((cf_sx1, cf_sx), axis = 0), np.concatenate((cf_sy1, cf_sy), axis = 0)

        Query = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        Test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

        query_loader = DataLoader(Query, batch_size  = batch_size)
        test_loader = DataLoader(Test, batch_size = batch_size)

        model = create_model(x_query.shape[1])
        optimizer = optim.Adam(model.parameters(), lr = 0.005)
        criterion = nn.BCELoss()
        device = torch.device("cpu")
        model = model.to(device)
        criterion = criterion.to(device)

        best_model = None
        best_acc = -float('inf')

        for epoch in range(500):

            train_loss, train_acc = train(model, query_loader, optimizer, criterion, device)
            '''
            valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

            if valid_acc > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = valid_acc
            '''
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} |')
                #print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |')

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Test acc: {test_acc:.4f}')
        acc_list.append(np.round(test_acc, 3))

    print(acc_list)
    print(np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3))

