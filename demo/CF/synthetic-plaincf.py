#Filename:	synthetic-plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 28 Jul 2021 02:58:48 

from nn import NNModel
import torch
from plainCF import PlainCF
from sklearn.preprocessing import MinMaxScaler
from Dataset import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dquery = pd.read_csv("../../data/synthetic/synthetic_query.csv").iloc[:, 0:2].to_numpy()
    dtrain = pd.read_csv("../../data/synthetic/synthetic_train.csv")
    
    ddtrain = dtrain.to_numpy().astype(np.float32)
    train_x, train_y = ddtrain[:, 0:2], ddtrain[:, 2:3]
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)


    d = Dataset(dataframe = dtrain, continuous_features = ['X1', 'X2'], outcome_name = 'Y', scaler = MinMaxScaler())
    features_to_vary = ['X1', 'X2']
    clf = NNModel(model_path = '../../data/synthetic/synthetic.pt')
    cf = PlainCF(d, clf)
    ddquery = d.scaler.transform(dquery)
    
    synthetic_cf = np.zeros((len(ddquery), 2))

    points = 3

    for i in range(len(ddquery)):
        print(i)
        #test_instance = ddquery[i:i+1]
        test_instance = ddquery[points:points+1]
        points_in_path = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary, _lambda = 1, lr = 0.0004, max_iter = 5000)
    
        break

    fig, ax = plt.subplots(figsize = (10, 10))
    h = 0.02
    x_min, x_max = ddtrain[:, 0].min() - .5, ddtrain[:, 0].max() + .5
    y_min, y_max = ddtrain[:, 1].min() - .5, ddtrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    model = torch.load("../../data/synthetic/synthetic.pt")
    dplot = np.c_[xx.ravel(), yy.ravel()]
    _dplot = scaler.transform(dplot).astype(np.float32)
    _dplot = torch.from_numpy(_dplot)
    #S = torch.round(model(_dplot)).detach().numpy()
    S = model(_dplot).detach().numpy()
    S = S.reshape(xx.shape)

    CS = ax.contour(xx, yy, S)
    ax.clabel(CS, fontsize=9, inline=1)
    ax.scatter(dquery[points, 0], dquery[points, 1])

    for i in range(len(points_in_path)-1):
        start = points_in_path[i]
        end = points_in_path[i+1]
        ax.annotate('', xy= end, xytext= start,
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1}, va='center', ha='center')
    
    plt.show()
