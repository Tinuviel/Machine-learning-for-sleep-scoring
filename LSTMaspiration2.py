#Eventuellt återkomma till detta om vi fouriertransformerar datan
import keras
from keras.models import Sequential
import scipy.io
from keras.losses import mean_absolute_percentage_error
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import random as rn
import os

#rat3 81% vs 61%, rat5 (4k = 43% vs 30%), (10k = 53% vs 65%), rat6 57% vs 35%, rat7 69% vs 57%, rat8 50% vs 30%
matData = scipy.io.loadmat('rat3_all.mat')
matrixA = matData['EEGandEMG']
matrixB = matData['labels'] 
matrixRowSize = matrixA.shape[1]

A = matrixA
B = matrixB
A = A.T
B = B.T
train_len = 10000
x_train = A[0:train_len,:]
y_train = B[0:train_len,:]
x_test = A[train_len:matrixRowSize,:]
y_test = B[train_len:matrixRowSize,:]

x_train = np.asarray(x_train).flatten()
y_train = np.asarray(y_train).flatten()
x_train = x_train.reshape(train_len, 1, 4000)
y_train = y_train.reshape(train_len, 1, 6)
x_test = np.asarray(x_test).flatten()
x_test = x_test.reshape(matrixRowSize-train_len, 1, 4000)
y_test =  y_test.reshape(matrixRowSize-train_len, 1, 6)

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, LSTM, regularizers
from keras.optimizers import SGD as SGD
from keras.optimizers import Adam

print('LSTMaspiration2')
model = Sequential()
model.add(LSTM(6, return_sequences = True, input_shape = (1, 4000), activation = 'softsign'))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(LSTM(6, return_sequences = True, input_shape = (1, 4000), activation = 'softsign'))
model.add(Dense(128, activation = 'relu'))
model.add(LSTM(6, return_sequences = True, input_shape = (1, 4000), activation = 'softsign'))

model.add(Dense(512))
model.add(Dense(units = 6, activation = 'softmax'))

optimizer = Adam(lr = 0.00001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, verbose = 2)

loss_and_metricsTRAIN = model.evaluate(x_train, y_train, batch_size = 64)
loss_and_metricsTEST = model.evaluate(x_test, y_test, batch_size = 64)
print('loss_and_metricsTRAIN ')
print(loss_and_metricsTRAIN)
print('loss_and_metricsTEST ')
print(loss_and_metricsTEST )
