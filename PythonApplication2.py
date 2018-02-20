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
train_len = 5000
x_train = A[0:train_len,:]
y_train = B[0:train_len,:]
x_test = A[train_len:19800,:]
y_test = B[train_len:19800,:]


#model = Sequential()

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD as SGD
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
arrA = np.asarray(x_train).flatten()
arrB = np.asarray(y_train).flatten()
arrA = arrA.reshape(train_len, 1, 4000)
target = y_train.reshape(train_len, 1, 6)
xtest = np.asarray(x_test).flatten()
xtest = xtest.reshape(19800-train_len, 1, 4000)
ytest =  y_test.reshape(19800-train_len, 1, 6)

model = Sequential()
model.add(LSTM(6, return_sequences=True, input_shape=(1, 4000), activation='softmax'))
model.add(LSTM(6, return_sequences=True, input_shape=(1, 4000), activation='softmax'))
model.add(LSTM(6, return_sequences=True, input_shape=(1, 4000), activation='softmax'))


model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Dense(units=6, activation='softmax'))

optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['categorical_accuracy', 'acc'])
model.fit(arrA, target, nb_epoch=10, batch_size=64, verbose=2)


loss_and_metrics = model.evaluate(xtest, ytest, batch_size=64)
print(loss_and_metrics)