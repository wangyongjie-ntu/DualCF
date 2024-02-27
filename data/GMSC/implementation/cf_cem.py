#Filename:	cf_gen.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Okt 2021 11:25:05 

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.datasets import load_boston
from alibi.explainers.cfproto import CounterfactualProto
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())

boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names

y = np.zeros((target.shape[0],))
y[np.where(target > np.median(target))[0]] = 1

data = np.delete(data, 3, 1)
feature_names = np.delete(feature_names, 3)

random.seed(0)
a = list(range(len(data)))
random.shuffle(a)
length = len(a)

train_x, train_y = data[a[0:int(0.5*length)]], y[a[0:int(0.5*length)]]
query_x, query_y = data[a[int(0.5*length):int(0.75*length)]], y[a[int(0.5*length):int(0.75*length)]]
test_x, test_y = data[a[int(0.75*length):]], y[a[int(0.75*length):]]

scaler = StandardScaler()
strain_x = scaler.fit_transform(train_x)
squery_x = scaler.transform(query_x)
stest_x = scaler.transform(test_x)

otrain_y = to_categorical(train_y)
oquery_y = to_categorical(query_y)
otest_y = to_categorical(test_y)

nn = load_model('nn_boston.h5')
print(nn.evaluate(squery_x, oquery_y))
print(nn.evaluate(stest_x, otest_y))

beta = .1  # L1
gamma = 10.  # autoencoder
theta = .1  # prototype

X = squery_x[1].reshape((1,) + squery_x[1].shape)
shape = X.shape

ae = load_model('boston_ae.h5')
enc = load_model('boston_enc.h5', compile=False)

cf = CounterfactualProto(nn, shape,
                         beta=beta,
                         enc_model=enc,
                         ae_model=ae,
                         gamma=gamma,
                         theta=theta,
                         max_iterations=1000,
                         feature_range= (strain_x.min(axis = 0), strain_x.max(axis = 0)),
                         c_init= 1.,
                         c_steps= 10
                        )

cf.fit(strain_x)

query_cf = np.zeros_like(squery_x)
query_cf_y = np.zeros(len(squery_x))
for idx in range(len(squery_x)):
    print(idx)
    X = squery_x[idx].reshape((1,) + squery_x[idx].shape)
    explanation = cf.explain(X)
    query_cf[idx] = explanation.cf['X']
    query_cf_y[idx] = explanation.cf['class']

tmp = np.concatenate((scaler.inverse_transform(query_cf), query_cf_y[:, np.newaxis]), axis = 1)
np.save("boston_housing_query_ae_cf.npy", tmp)

query_ccf = np.zeros_like(squery_x)
query_ccf_y = np.zeros(len(squery_x))

for idx in range(len(query_cf)):
    print(idx)
    X = query_cf[idx].reshape((1,) + query_cf[idx].shape)
    explanation = cf.explain(X)
    query_ccf[idx] = explanation.cf['X']
    query_ccf_y[idx] = explanation.cf['class']

tmp1 = np.concatenate((scaler.inverse_transform(query_ccf), query_ccf_y[:, np.newaxis]), axis = 1)
np.save("boston_housing_query_ae_2cf.npy", tmp1)
