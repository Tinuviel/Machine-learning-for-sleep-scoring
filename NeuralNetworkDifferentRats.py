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

#rat3 63%, rat5 60%, rat6 53%, rat7 59%, rat8 34%
matData = scipy.io.loadmat('rat3_allextraSamp.mat')
matData2 = scipy.io.loadmat('rat5_allSTDsplit.mat')
matrixA3 = matData['EEGandEMG']
matrixA5 = matData2['EEGandEMG']
matrixB3 = matData['labels'] 
matrixB5 = matData2['labels']
matrixRowSize = matrixA3.shape[1]
A3 = matrixA3
B3 = matrixB3
A3 = A3.T
B3 = B3.T
A5 = matrixA5
B5 = matrixB5
A5 = A5.T
B5 = B5.T
x_train = A3
y_train = B3
x_test = A5
y_test = B5

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
class_weight = {0 : 1.,
                1: 0.,
                2: 1,
                3: 0,
                4: 7,
                5: 0}
print('NeuralNetworkDifferentRats')
model = Sequential()
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_dim=4000))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
BatchNormalization(axis=1)
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#Either use validation_data = validation_set with validation_set initialized, OR use validation_split = some value between 0 and 1
model.fit(x_train, y_train, epochs=20, batch_size=512, shuffle = True, class_weight = class_weight, verbose=2)

#model.fit(x_train, y_train, epochs=20, batch_size=128, shuffle = True, validation_split = 0.2, class_weight = class_weight, verbose=2)
score1 = model.evaluate(x_train, y_train, batch_size=512)
score = model.evaluate(x_test, y_test, batch_size=512)
print(score1)
print(score)

from sklearn.metrics import classification_report
y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(y_test, y_pred))