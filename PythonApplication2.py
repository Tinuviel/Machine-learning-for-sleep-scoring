import keras
from keras.models import Sequential
import scipy.io
from keras.losses import mean_absolute_percentage_error
import numpy as np

matData = scipy.io.loadmat('rat3_all.mat')
matrixA = matData['EEGandEMG']
matrixB = matData['labels'] 

L = matrixA.shape
print(L)

A = matrixA
B = matrixB
A = A.T
B = B.T
x_train = A[0:15000,:]
y_train = B[0:15000,:]
x_test = A[15000:19800,:]
y_test = B[15000:19800,:]


#model = Sequential()

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD as SGD
from keras.utils.np_utils import to_categorical
arrA = np.asarray(x_train).flatten()
arrB = np.asarray(y_train).flatten()
arrA = arrA.reshape(15000, 1, 4000)
target = y_train.reshape(15000, 1, 6)

#model.add(Reshape(1, 4000, 256), input_shape=4000)
#model.add(Convolution1D(10, 10))

#model.add(Convolution1D(32, 3, activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))

#model.add(Dense(units=64, activation='relu', input_dim=4000))
#Denna funkar inte just nu
#model.add(LSTM(128, input_shape=4000))
#model.add(Dense(128, activation='linear'))
#model.add(Dropout(0.3))
#model.add(Dense(256))
#model.add(Dense(32))
#model.add(Dropout(0.1))
batch_size = 1
model = Sequential()
model.add(LSTM(6, return_sequences=True, input_shape=(1, 4000), activation='sigmoid'))
model.add(LSTM(6, return_sequences=True, input_shape=(1, 4000), activation='sigmoid'))
model.add(Dropout(0.2))
#model.add(Dense(6))
#model.compile(loss='mean_squared_error', optimizer='adam')


model.add(Dense(units=6, activation='softmax'))
optimizer = SGD(lr = 0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
model.fit(arrA, target, nb_epoch=100, batch_size=50, verbose=2)
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
              
#model.fit(x_train, y_train, epochs=12, batch_size=512)

#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=512)
#print(loss_and_metrics)