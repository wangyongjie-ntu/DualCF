#Filename:	plainCF.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 13 Des 2020 09:15:05  WIB

import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class PlainCF(object):

    def __init__(self, model_interface):

        self.model_interface = model_interface

    def generate_counterfactuals(self, query_instance, cur_prediction, cf_initialize, scaler, maxx, minx, _lambda = 10,
            optimizer = "adam", lr = 0.1, max_iter = 1000, target = None, mads = None):
       
        start_time = time.time()
        query_instance = torch.FloatTensor(query_instance)
        minx = torch.FloatTensor(minx)
        maxx = torch.FloatTensor(maxx)
        self._lambda = _lambda

        if mads is not None:
            self.mads = torch.FloatTensor(mads)
        else:
            self.mads = mads

        self.cur_prediction = cur_prediction

        if target is None:
            if self.cur_prediction == 0:
                target = 0.6
            else:
                target = 0.4
        
        cf_initialize = torch.from_numpy(cf_initialize)
        rand = torch.randn(query_instance.shape)
        cf_initialize = cf_initialize + rand * 0.1
        cf_initialize = torch.FloatTensor(cf_initialize)

        cf_initialize.requires_grad_(True)

        if optimizer == "adam":
            optim = torch.optim.Adam([cf_initialize], lr)
        else:
            optim = torch.optim.RMSprop([cf_initialize], lr)

        for i in range(max_iter):
            optim.zero_grad()
            loss = self.compute_loss(cf_initialize, query_instance, target)
            loss.backward()
            optim.step()
            
            if isinstance(scaler, StandardScaler):
                cf_initialize.data = torch.where(cf_initialize.data > maxx, maxx, cf_initialize.data)
                cf_initialize.data = torch.where(cf_initialize.data < minx, minx, cf_initialize.data)

        end_time = time.time()
        running_time = time.time()
        return cf_initialize.data.numpy()

    def compute_loss(self, cf_initialize, query_instance, target):
        
        if self.cur_prediction == 0:
            loss1 = F.relu(target - self.model_interface.predict_tensor(cf_initialize)[1])
        else:
            loss1 = F.relu(self.model_interface.predict_tensor(cf_initialize)[1] - target)
        loss2 = torch.sum((cf_initialize - query_instance)**2)
        #loss2 = torch.sum(torch.abs(cf_initialize - query_instance) / self.mads)
        return self._lambda * loss1 + loss2

