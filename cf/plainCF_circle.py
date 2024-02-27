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

    def __init__(self, data_interface, model_interface):

        self.data_interface = data_interface
        self.model_interface = model_interface

    def generate_counterfactuals(self, query_instance, _lambda = 10, optimizer = "adam", lr = 0.1, max_iter = 1000):
       
        start_time = time.time()
        if isinstance(query_instance, dict) or isinstance(query_instance, list):
            query_instance = self.data_interface.prepare_query(query_instance, normalized = True)

        query_instance = torch.FloatTensor(query_instance)
        self._lambda = _lambda
        self.cur_prediction = self.model_interface.predict_tensor(query_instance)[0].item()

        if self.cur_prediction == 0:
            target = 0.9
            cf_initialize = torch.tensor([[0.5, 0.5]]).float()
            rand = torch.randn(query_instance.shape)
            cf_initialize = cf_initialize + rand * 0.1
        else:
            target = 0.1
            center = torch.tensor([[0.5, 0.5]]).float()
            cf_initialize = center + 3 * (query_instance - center)
            rand = torch.randn(query_instance.shape)
            cf_initialize = cf_initialize + 0.1 * rand

        cf_initialize = torch.FloatTensor(cf_initialize)
        cf_initialize.requires_grad_(True)

        if optimizer == "adam":
            optim = torch.optim.Adam([cf_initialize], lr)
        else:
            optim = torch.optim.RMSprop([cf_initialize], lr)

        for i in range(max_iter):
            cf_initialize.requires_grad = True
            optim.zero_grad()
            loss = self.compute_loss(cf_initialize, query_instance, target)
            loss.backward()
            optim.step()
            
            if isinstance(self.data_interface.scaler, MinMaxScaler):
                cf_initialize.data = torch.where(cf_initialize.data > 1, torch.ones_like(cf_initialize.data), cf_initialize.data)
                cf_initialize.data = torch.where(cf_initialize.data < 0, torch.zeros_like(cf_initialize.data), cf_initialize.data)


        end_time = time.time()
        running_time = time.time()

        return cf_initialize.data.numpy()

    def compute_loss(self, cf_initialize, query_instance, target):
        
        if self.cur_prediction == 0:
            loss1 = F.relu(target - self.model_interface.predict_tensor(cf_initialize)[1])
        else:
            loss1 = F.relu(self.model_interface.predict_tensor(cf_initialize)[1] - target)

        loss2 = torch.sum((cf_initialize - query_instance)**2)
        return self._lambda * loss1 + loss2

