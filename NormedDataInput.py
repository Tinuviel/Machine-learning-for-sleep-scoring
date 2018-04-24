import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
import random as rn
import scipy.io
import numpy as np
import os


#Same random seed is used everytime for reproducibility
#Only one thread
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

matData = scipy.io.loadmat('rat7_allSTDfft.mat')
matrixA = matData['fftEEGandEMG']
matrixB = matData['labels'] 
matrixRowSize = matrixA.shape[1]
A = matrixA
B = matrixB
A = A.T
B = B.T
train_len = 7000
val_len = 1000
x_train = A[0:train_len,:]
y_train = B[0:train_len,:]
x_val = A[train_len:train_len+val_len, :]
y_val = B[train_len:train_len+val_len, :]
x_test = A[train_len+val_len:matrixRowSize,:]
y_test = B[train_len+val_len:matrixRowSize,:]

index = [0] * matrixRowSize
for i in range(0, matrixRowSize):
    index[i] = i

#Initialize training sets with the first train_len random rows
for row in range(0, train_len):
    test = np.random.randint(0, matrixRowSize-1)
    x_train[row, :] = A[index[test], :]
    y_train[row, :] = B[index[test], :]
    index.remove(index[test])
    matrixRowSize = matrixRowSize-1

#Initialize validation sets with the next val_len random rows
for row in range(0, val_len):
    test = np.random.randint(0, matrixRowSize-1)
    x_val[row, :] = A[index[test], :]
    y_val[row, :] = B[index[test], :]
    index.remove(index[test])
    matrixRowSize = matrixRowSize-1

#Initialize testing sets with the rest of the rows
for i in range(0, len(index)):
    x_test[i, :] = A[index[i], :]
    y_test[i, :] = B[index[i], :]

from keras.layers import Dense
from keras import losses 
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, regularizers
from keras.optimizers import SGD
from sklearn.utils import class_weight
from keras.layers.normalization import BatchNormalization
#This should give class_weights that are balanced from the dataset, but i get 'numpy.ndarray is unhashable'-error
#class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#class_weight_dict = dict(enumerate(class_weight))

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn

#Weighting classes, ANY weights except 1 on everyone gives lower accuracy
#0 = W, 1 = X, 2 = S, 3 = 1, 4 = P, 5 = 2
class_weight = {0 : 1,
                1: 0,
                2: 1,
                3: 0,
                4: 1,
                5: 0}
print('NormedDataInput')
model = Sequential()
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1), input_dim=4000))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
BatchNormalization(axis=1)
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#Either use validation_data = validation_set with validation_set initialized, OR use validation_split = some value between 0 and 1
model.fit(x_train, y_train, epochs=40, batch_size=512, shuffle = True, class_weight = class_weight, verbose=2)

#model.fit(x_train, y_train, epochs=20, batch_size=128, shuffle = True, validation_split = 0.2, class_weight = class_weight, verbose=2)
score1 = model.evaluate(x_train, y_train, batch_size=512)
score = model.evaluate(x_test, y_test, batch_size=512)
print(score1)
print(score)

from sklearn.metrics import classification_report
y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(y_test, y_pred))