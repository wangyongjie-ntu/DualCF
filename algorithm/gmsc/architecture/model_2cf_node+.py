#Filename:	gmsc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 12 Jul 2021 10:43:57 

import torch
import sys
import copy
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, Dataset

class MyDataset(Dataset):

    def __init__(self, query_x, cf_x, query_y, cf_y):
        super()
        self.query_x = query_x
        self.cf_x = cf_x
        self.query_y = query_y
        self.cf_y = cf_y

    def __len__(self):
            return len(self.cf_x)
    
    def __getitem__(self, index):
        query_x = self.query_x[index]
        cf_x = self.cf_x[index]
        query_y = self.query_y[index]
        cf_y = self.cf_y[index]

        train_x = np.concatenate((query_x, cf_x), axis = 0)
        train_y = np.concatenate((query_y, cf_y), axis = 0)
    
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        
        return train_x, train_y

def train(model, train_loader, optimizer, criterion, device):
    epoch_loss = 0
    prediction = []
    label = []
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = data.reshape(-1, 10), target.reshape(-1, 1)
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
    
    dquery = np.load("../../../data/GMSC/gmsc_query.npy").astype(np.float32)
    dtest = np.load("../../../data/GMSC/gmsc_test.npy").astype(np.float32)
    dcf = np.load("../../../data/GMSC/64/gmsc-plaincf-l2.npy").astype(np.float32)
    dccf = np.load("../../../data/GMSC/64/gmsc-plaincf2-l2.npy").astype(np.float32)

    query_x, query_y = dquery[:, 1:], dquery[:, 0:1]
    cf_x, cf_y = dcf[:, 1:], dcf[:, 0:1]
    ccf_x, ccf_y = dccf[:, 1:], dccf[:, 0:1]
    test_x, test_y = dtest[:, 1:], dtest[:, 0:1]

    scaler = StandardScaler()
    query_x = scaler.fit_transform(query_x)
    test_x = scaler.transform(test_x)
    cf_x = scaler.transform(cf_x)
    ccf_x = scaler.transform(ccf_x)

    input_len = query_x.shape[1]
    a = list(range(len(query_x)))
    length = int(sys.argv[1])
    if length == 0:
        length = len(a)
    
    print(length)
    acc_list = []
    
    for i in range(100):
        print(i)
        random.shuffle(a)
        idx = a[0:length]

        ccf_sx, ccf_sy = ccf_x[idx], ccf_y[idx]
        cf_sx, cf_sy = cf_x[idx], cf_y[idx]

        Query = MyDataset(ccf_sx, cf_sx, ccf_sy, cf_sy)
        Test = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        query_loader = DataLoader(Query, batch_size = 128)
        test_loader = DataLoader(Test, batch_size = 128)

        model = nn.Sequential(
                nn.Linear(input_len, 20),
                nn.ReLU(),
                nn.Linear(20, 15),
                nn.ReLU(),
                nn.Linear(15, 1),
                nn.Sigmoid(),
                )

        device = torch.device("cpu")

        criterion = nn.BCELoss()
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr = 0.005)

        best_model = None
        best_acc = -float('inf')

        for epoch in range(200):
            train_loss, train_acc = train(model, query_loader, optimizer, criterion, device)

            if epoch % 50 == 0:
                print(train_loss, train_acc)
        
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f'Test Acc: {test_acc:.4f}')
        acc_list.append(np.round(test_acc, 3))

    print(acc_list)
    print(np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3))
