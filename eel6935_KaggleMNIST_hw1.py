
from __future__ import print_function
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adagrad, Adam, RMSprop

batch_size = 128
nb_classes = 10
nb_epoch = 50

# input dimensions
img_rows, img_cols = 28, 28

# number of convolutional filters
nb_filters = 32

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

trainfile = pd.read_csv('/home/vivek/eel6935/data/train.csv')
testfile = pd.read_csv('/home/vivek/eel6935/data/test.csv')

trainfile_X = trainfile.iloc[:,1:]
#print (trainfile_X)

testfile_X = testfile.iloc[:,:]
#print (np.asarray(testfile_X).shape)

X_train = np.asarray(trainfile_X)
#print (type(X_train))

X_test = np.asarray(testfile_X)
#print (type(X_test))

#print (X_train.shape)
#print (X_test.shape)

trainfile_Y = trainfile.iloc[:,:1]

#print (trainfile_Y)

y_train = np.asarray(trainfile_Y)
#print (y_train.shape)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print('X_test shape:', X_test.shape)
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('tanh'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

model.summary()

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

optimizer = RMSprop(lr=0.007)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
score = model.predict_classes(X_test, batch_size=batch_size, verbose=0)

target = open('output_adam_2tanh_rmsprop_0_007_0220_1.csv','w')

target.write('imageid,label\n')
for i in range(10000):
    target.write('%s,%s\n' % (i+1,score[i]))
    

target.close()

