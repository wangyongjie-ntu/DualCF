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
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

class TripletDataset(data.Dataset):

    def __init__(self, dquery_x, dquery_y, dcf_x, dcf_y, perturb_method = None, perturb_num = 200):
        super().__init__()
        self.dquery_x, self.dquery_y = dquery_x, dquery_y
        self.dcf_x, self.dcf_y = dcf_x, dcf_y
        self.perturb_num = perturb_num
        self.perturb_method = perturb_method

    def __len__(self):
        return len(self.dquery_x)

    def __getitem__(self, index):
        
        query_x, query_y = self.dquery_x[index], self.dquery_y[index]
        cf_x, cf_y = self.dcf_x[index], self.dcf_y[index]
        query_x = query_x.repeat(self.perturb_num).reshape((self.perturb_num, 2))
        cf_x = cf_x.repeat(self.perturb_num).reshape((self.perturb_num, 2))
        query_y = query_y.repeat(self.perturb_num).reshape((self.perturb_num, 1))
        cf_y = cf_y.repeat(self.perturb_num).reshape((self.perturb_num, 1))
    
        if self.perturb_method == "mask":
            mask = torch.randint(0, 2, (self.perturb_num, 2))
            base_x = cf_x * mask

        elif self.perturb_method == "noise":
            noise = torch.randn(self.perturb_num, 2)
            base_x = 0.1 * noise + cf_x
        elif self.perturb_method is None:
            base_x  = torch.rand(self.perturb_num, 2)

        return query_x, cf_x, base_x, query_y, cf_y

class CFJointLoss(nn.Module):

    def __init__(self, lambda_ = 0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.loss = nn.BCELoss()

    def euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def cls_loss(self, pred1, pred2, target1, target2):
        return self.loss(pred1, target1) + self.loss(pred2, target2) + 1e-8

    def forward(self, query_x, cf_x, base_x, pred_x, pred_cf, pred_base, target_x, target_cf):

        distance1 = (pred_cf - pred_x).squeeze().abs() / self.euclidean(query_x, cf_x)
        distance2 = (pred_base - pred_x).squeeze().abs() / self.euclidean(query_x, base_x)
        triplet_loss = F.relu(distance2 - distance1).mean()
        classification_loss = self.cls_loss(pred_x, pred_cf, target_x, target_cf)
        total_loss = classification_loss + self.lambda_ * triplet_loss
        return total_loss, classification_loss, triplet_loss

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    cls_loss = 0
    t_loss = 0
    model.train()
    prediction = []
    label = []

    for batch_idx, (query_x, cf_x, base_x, query_y, cf_y) in enumerate(iterator):
        query_x, cf_x, base_x, query_y, cf_y = query_x.to(device), cf_x.to(device), base_x.to(device), query_y.to(device), cf_y.to(device)
        query_x = query_x.reshape((-1, query_x.shape[-1]))
        cf_x = cf_x.reshape((-1, cf_x.shape[-1]))
        base_x = base_x.reshape((-1, base_x.shape[-1]))
        query_y = query_y.reshape((-1, query_y.shape[-1]))
        cf_y = cf_y.reshape((-1, cf_y.shape[-1]))
        optimizer.zero_grad()
        output1 = model(query_x)
        output2 = model(cf_x)
        output3 = model(base_x)
        loss, classification_loss, triplet_loss = criterion(query_x, cf_x, base_x, output1, output2, output3, query_y, cf_y)
        loss.backward()
        optimizer.step()
        preds = torch.round(output1)
        epoch_loss += loss.item() * len(query_x)
        cls_loss += classification_loss.item() * len(query_x)
        t_loss += triplet_loss.item() * len(query_x)
        label.extend(query_y.reshape(-1).tolist())
        prediction.extend(preds.reshape(-1).tolist())

    acc = accuracy_score(label, prediction)

    return epoch_loss / (len(iterator.dataset) * 10), cls_loss/ (len(iterator.dataset) * 10), t_loss / (len(iterator.dataset) * 10), acc

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

    dquery = np.load("../../data/synthetic_v1/synthetic_query_v1.npy").astype(np.float32)
    dval = np.load("../../data/synthetic_v1/synthetic_val_v1.npy").astype(np.float32)
    dtest = np.load("../../data/synthetic_v1/synthetic_test_v1.npy").astype(np.float32)
    dcf = np.load("../../data/synthetic_v1/synthetic-plaincf.npy").astype(np.float32)

    query_x, query_y = dquery[:, 0:2], dquery[:, 2:3]
    cf_x, cf_y = dcf[:, 0:2], np.round(1 - query_y).astype(np.float32)
    val_x, val_y = dval[:, 0:2], dval[:, 2:3]
    test_x, test_y = dtest[:, 0:2], dtest[:, 2:3]

    scaler = MinMaxScaler()
    query_x = scaler.fit_transform(query_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    cf_x = scaler.transform(cf_x)

    query_x, val_x, test_x, cf_x = torch.from_numpy(query_x), torch.from_numpy(val_x), torch.from_numpy(test_x), torch.from_numpy(cf_x)
    query_y, val_y, test_y, cf_y = torch.from_numpy(query_y), torch.from_numpy(val_y), torch.from_numpy(test_y), torch.from_numpy(cf_y)

    a = list(range(len(query_x)))
    length = int(sys.argv[1])
    if length == 0:
        length = len(a)
    print(length)
    acc_list = []

    for seed in range(30):
        print("seed used :{}".format(seed))
        random.seed(seed)
        torch.manual_seed(seed)
        random.shuffle(a)
        idx = a[0:length]
        query_sx, query_sy = query_x[idx], query_y[idx]
        cf_sx, cf_sy = cf_x[idx], cf_y[idx]

        Query = TripletDataset(query_sx, query_sy, cf_sx, cf_sy, perturb_method = 'noise')
        Val = TensorDataset(val_x, val_y)
        Test = TensorDataset(test_x, test_y)

        query_loader = DataLoader(Query, batch_size  = 32)
        val_loader = DataLoader(Val, batch_size = 32)
        test_loader = DataLoader(Test, batch_size = 32)

        model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )


        optimizer = optim.Adam(model.parameters(), lr = 0.005)
        criterion1 = nn.BCELoss()
        criterion2 = CFJointLoss(lambda_ = 0.05)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model = model.to(device)
        criterion1 = criterion1.to(device)
        criterion2 = criterion2.to(device)

        best_model = None
        best_acc = -float('inf')

        for epoch in range(500):

            train_loss, cls_loss, t_loss, train_acc = train(model, query_loader, optimizer, criterion2, device)
            '''
            valid_loss, valid_acc = evaluate(model, test_loader, criterion1, device)

            if valid_acc > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = valid_acc
            '''
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} |')
                print(f'Epoch: {epoch} | Classification Loss: {cls_loss:.4f} |Triplet Loss: {t_loss:.4f} |')
                #print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |')

        test_loss, test_acc = evaluate(model, test_loader, criterion1, device)
        print(f'Test acc: {test_acc:.4f}')
        acc_list.append(test_acc)

    print(acc_list)
    print(np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3))
