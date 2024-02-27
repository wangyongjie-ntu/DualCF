#Filename:	train.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 05 Des 2020 02:28:30  WIB

import torch
import time

def train_query(model, train_loader, loss_func, optimizer, is_gpu = True):

    model.train()
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0

    for step, (img, label) in enumerate(train_loader):

        if is_gpu:
            img = img
            label = label
        
        optimizer.zero_grad()
        pred = model(img) 
        _, out = torch.max(pred, 1)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(out == label).item()

    end = time.time()
    time_elapsed = end - start
    return model, epoch_loss, epoch_acc, time_elapsed

def train_counterquery(model, train_loader, loss_func, optimizer, is_gpu = True):

    model.train()
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0

    for step, (img1, img2, label1, label2) in enumerate(train_loader):

        if is_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
            label1, label2 = label1.long().cuda(), label2.long().cuda()

        
        optimizer.zero_grad()
        pred1, pred2 = model(img1), model(img2)
        _, out1 = torch.max(pred1, 1)
        _, out2 = torch.max(pred2, 1)
        loss = loss_func(pred1, pred2, label1, label2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(out1 == label1).item()
        epoch_acc += torch.sum(out2 == label2).item()

    end = time.time()
    time_elapsed = end - start
    return model, epoch_loss, epoch_acc, time_elapsed

def train_countertriplet(model, train_loader, loss_func, optimizer, is_gpu = True):

    model.train()
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0

    for step, (img1, img2, img3, label1, label2) in enumerate(train_loader):

        if is_gpu:
            img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
            label1, label2 = label1.long().cuda(), label2.long().cuda()
        
        optimizer.zero_grad()
        pred1, pred2, pred3 = model(img1), model(img2), model(img3)
        _, out1 = torch.max(pred1, 1)
        _, out2 = torch.max(pred2, 1)
        loss = loss_func(img1, img2, img3, pred1, pred2, pred3, label1, label2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(out1 == label1).item()
        epoch_acc += torch.sum(out2 == label2).item()

    end = time.time()
    time_elapsed = end - start
    return model, epoch_loss, epoch_acc, time_elapsed

def test(model, test_loader, loss_func, is_gpu = True):

    start = time.time()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        model.eval()

        for step, (batch_x, batch_y) in enumerate(test_loader):
            if ig_gpu:
                img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
                label1, label2 = label1.long().cuda(), label2.long().cuda()

        
            pred1, pred2, pred3 = model(img1), model(img2), model(img3)
            _, out1 = torch.max(pred1, 1)
            _, out2 = torch.max(pred2, 1)
            loss = loss_func(img1, img2, img3, pred1, pred2, pred3, label1, label2)
            epoch_loss += loss.detach().item()
            epoch_acc += torch.sum(out1 == label1).item()
            epoch_acc += torch.sum(out2 == label2).item()


    end = time.time()
    time_elapsed = end - start
    return model, epoch_loss, epoch_acc, time_elapsed
    
