#Filename:	mnist.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 04 Des 2020 01:15:15  WIB

import torch.nn.functional as F
import torch.nn as nn
import torch

class JointLoss(nn.Module):
    def __init__(self, margin, lambda_):
        super().__init__()
        self .margin = margin
        self.lambda_ = lambda_
        self.loss = nn.CrossEntropyLoss()

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum([1, 2])

    def cross_entropy_loss(self, pred1, pred2, target1, target2):
        return self.loss(target1, pred1) + self.loss(target2, pred2)

    def forward(self, img1, img2, img3, pred1, pred2, pred3, target1, target2):
        # target2: batch_size * 1
        distance1 = (torch.gather(pred2, 1, target2.unsqueeze(-1)) - torch.gather(pred1, 1, target2.unsqueeze(-1))) / self.calc_euclidean(img1, img2)
        distance2 = (torch.gather(pred3, 1, target2.unsqueeze(-1)) - torch.gather(pred1, 1, target2.unsqueeze(-1))) / self.calc_euclidean(img1, img3)
        triplet_loss = F.relu(distance1 - distance2 + self.margin)
        classification_loss = self.cross_entropy_loss(pred1, pred2, target1, target2)
        total_loss = classification_loss + self.lambda_ * triplet_loss
        return total_loss.mean()

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, target1, target2):
        return self.loss(pred1, target1) + self.loss(pred2, target2)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        X = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = X.view(-1, 576)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features * s
        return num_features

