#Filename:	loader.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 05 Des 2020 12:17:27  WIB

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
import os
import random

class CounterTripletDataset(data.Dataset):
    def __init__(self, file_list, transform = None, perturbation = "blur", perturb_num = 5):

        self.file_list = file_list
        self.transform = transform
        self.perturb_num = perturb_num

    def __len__(self):
        return len(self.file_list) * self.perturb_num

    def __getitem__(self, index):
        index = index % len(self.file_list)
        radius = random.randint(1, self.perturb_num)
        imgpath1, imgpath2 = self.file_list[index]
        label1, label2 = get_label(imgpath1), get_label(imgpath2)
        img1 = Image.open(imgpath1).convert("L")
        img2 = Image.open(imgpath2).convert("L")
        img3 = img2.filter(ImageFilter.GaussianBlur(radius)) # blur with random radius
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
    
        return img1, img2, img3, label1, label2

class CounterQueryDataset(data.Dataset):

    def __init__(self, file_list, transform = None):

        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        imgpath1, imgpath2 = self.file_list[index]
        label1, label2 = get_label(imgpath1), get_label(imgpath2)
        img1 = Image.open(imgpath1).convert("L")
        img2 = Image.open(imgpath2).convert("L")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label1, label2

class QueryDataset(data.Dataset):
    def __init__(self, file_list, transform = None):

        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        imgpath1, _ = self.filelist[index]
        label = get_label(imgpath1)
        img = Image.open(imgpath1).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img1, label1


def load_file_list(filename):
    files = pd.read_csv(filename, sep = '\t', header = None, index_col = None)
    file_list = []
    for _, i in files.iterrows():
        file_list.append(i.to_list())
    
    return file_list

def get_label(filename):
    index1 = filename.rindex('/')
    index2 = filename[0:index1].rindex('/')
    return int(filename[index2+1:index1])

def train_loader(file_list, batch_size, dtype):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    if dtype == "query":
        querydataset = QueryDataset(file_list, transform = transform)
        train_loader = data.DataLoader(querydataset, batch_size = batch_size, shuffle = True, num_workers = 20)
    elif dtype == "counterquery":
        counterquerydataset = CounterQueryDataset(file_list, transform = transform)
        train_loader = data.DataLoader(counterquerydataset, batch_size = batch_size, shuffle = True, num_workders = 20)

    elif dtype == "countertriplet":
        countertripletdataset = CounterTripletDataset(file_list, transform = transform)
        train_loader = data.DataLoader(countertripletdataset, batch_size = batch_size, shuffle = True, num_workers = 20)

    return train_loader

def test_loader(file_list, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    querydataset = QueryDataset(file_list, transform = transform)
    test_loader = data.DataLoader(querydataset, batch_size = batch_size, shuffle = True, num_workers = 20)

    return test_loader

if __name__ == "__main__":
    file_list = load_file_list('./util/files.txt')
    loader = train_loader(file_list, 2, "countertriplet")
    for i, (img1, img2, img3, label1, label2) in enumerate(loader):
        print(img1.shape, img2.shape, img3.shape)
        break

