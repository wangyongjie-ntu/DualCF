#Filename:	model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Okt 2021 11:23:22 

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

def nn_model():
    x_in = Input(shape=(12,))
    x = Dense(40, activation='relu')(x_in)
    x = Dense(40, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn

nn = nn_model()
nn.summary()
nn.fit(strain_x, otrain_y, batch_size=64, epochs=500, verbose=0)
nn.save('nn_boston.h5', save_format='h5')
