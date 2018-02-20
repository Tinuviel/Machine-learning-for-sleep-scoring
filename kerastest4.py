import keras
from keras.models import Sequential
import scipy.io

matData = scipy.io.loadmat('rat3_all.mat')
matrixA = matData['EEGandEMG']
matrixB = matData['labels'] 

A = matrixA
B = matrixB
A = A.T
B = B.T
x_train = A[0:9900,:]
y_train = B[0:9900,:]
x_test = A[9900:19800,:]
y_test = B[9900:19800,:]


model = Sequential()

from keras.layers import Dense
from keras import losses 
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=4000))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
