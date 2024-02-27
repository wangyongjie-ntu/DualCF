#Filename:	heloc_cf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 04 Okt 2021 09:39:05 

from aix360.datasets.heloc_dataset import *
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense
from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random

heloc = HELOCDataset()
df = heloc.dataframe()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
print("Size of HELOC dataset:", df.shape)
print("Number of \"Good\" applicants:", np.sum(df['RiskPerformance']=='Good'))
print("Number of \"Bad\" applicants:", np.sum(df['RiskPerformance']=='Bad'))
print("Sample Applicants:")
df.head(10).transpose()

df['RiskPerformance'].replace(to_replace=['Bad', 'Good'], value=[0, 1], inplace=True)

random.seed(0)
a = list(range(len(df)))
random.shuffle(a)
length = len(a)

train = df.iloc[a[0:int(len(a) * 0.5)]]
query = df.iloc[a[int(len(a) * 0.5):int(len(a) * 0.75)]]
test = df.iloc[a[int(len(a) * 0.75):]]

train_x, train_y = train.iloc[:, 0:-1].values, train.iloc[:, -1].values
query_x, query_y = query.iloc[:, 0:-1].values, query.iloc[:, -1].values
test_x, test_y = test.iloc[:, 0:-1].values, test.iloc[:, -1].values

scaler = MinMaxScaler()
strain_x = scaler.fit_transform(train_x)
squery_x = scaler.fit_transform(query_x)
stest_x = scaler.fit_transform(test_x)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
query_y = to_categorical(query_y)

def nn_small():
    model = Sequential()
    model.add(Dense(10, input_dim=23, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))
    return model

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

# compile and print model summary
nn = nn_small()
nn.compile(loss=fn, optimizer='adam', metrics=['accuracy'])
nn.summary()

TRAIN_MODEL = True

if (TRAIN_MODEL):
    nn.fit(strain_x, train_y, batch_size=128, epochs=300, verbose=1, shuffle=False)
    nn.save_weights("heloc_nnsmall.h5")
else:
    nn.load_weights("heloc_nnsmall.h5")


# evaluate model accuracy
score = nn.evaluate(strain_x, train_y, verbose=0) #Compute training set accuracy
#print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = nn.evaluate(squery_x, query_y, verbose=0) #Compute test set accuracy
#print('Test loss:', score[0])
print('Query accuracy:', score[1])

score = nn.evaluate(stest_x, test_y, verbose=0) #Compute test set accuracy
#print('Test loss:', score[0])
print('Test accuracy:', score[1])


mymodel = KerasClassifier(nn)
explainer = CEMExplainer(mymodel)

arg_mode = 'PN' # Find pertinent negatives
arg_max_iter = 1000 # Maximum number of iterations to search for the optimal PN for given parameter settings
arg_init_const = 10.0 # Initial coefficient value for main loss term that encourages class change
arg_b = 9 # No. of updates to the coefficient of the main loss term
arg_kappa = 0.2 # Minimum confidence gap between the PNs (changed) class probability and original class' probability
arg_beta = 1e-1 # Controls sparsity of the solution (L1 loss)
arg_gamma = 100 # Controls how much to adhere to a (optionally trained) auto-encoder
my_AE_model = None # Pointer to an auto-encoder
arg_alpha = 0.01 # Penalizes L2 norm of the solution
arg_threshold = 1. # Automatically turn off features <= arg_threshold if arg_threshold < 1
arg_offset = 0.5 # the model assumes classifier trained on data normalized
                # in [-arg_offset, arg_offset] range, where arg_offset is 0 or 0.5
query_cf = np.zeros_like(query_x)
query_cf_y = np.zeros(len(query_y))

for i in range(len(query_x)):
    
    print("{} in all {} example".format(i, len(query_x)))
    X = squery_x[i].reshape((1,) + squery_x[i].shape)
    (cf, delta_cf, info_cf) = explainer.explain_instance(X, arg_mode, my_AE_model, arg_kappa, arg_b,
                                                         arg_max_iter, arg_init_const, arg_beta, arg_gamma,
                                                            arg_alpha, arg_threshold, arg_offset)
    query_cf[i] = cf
    query_cf_y[i] = np.argmax(nn.predict_proba(cf))
    print(query_cf_y[i])

tmp = np.concatenate((scaler.inverse_transform(query_cf), query_cf_y[:, np.newaxis]), axis = 1)
np.save("heloc_query_cf.npy", tmp)

