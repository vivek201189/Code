from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score
from metrics_04 import MyMetric

batch_size = 32
nb_classes = 10
nb_epoch = 10

# input dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def load_training_data():

    file1 = open("/home/vivek/ocr_merged_file.txt", "r")
    every = file1.readlines()

    digit = []
    mega = []
    labels = []

    i = 0
    while(i < len(every)-1):
        if(len(list(every[i])) > 3):
            a = list(every[i])
            a.pop()
            digit.append(a)
        if(len(list(every[i])) == 3):
            a = list(every[i])
            b = list(a[1])
            labels.append(b)
            mega.append(digit)
            digit = []
        i = i + 1

    X = np.array(mega[:1124], dtype=np.float)
    y = np.array(labels[:1124], dtype=np.int)
    file1.close()
    
    Xt = np.array(mega[1124:5620], dtype=np.float)
    yt = np.array(labels[1124:5620], dtype=np.int)
    
    return X, y, Xt, yt

def create_model():
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, cond):
    
    #Xt = Xt.reshape(Xt.shape[0], 1, img_rows, img_cols)
    #Xt = Xt.astype('float32')
    #Xt /= 255
    
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    
    score = model.evaluate(X_test, Y_test)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])
        
    classes = model.predict_classes(X_test, batch_size=32)
    prob = model.predict_proba(X_test, batch_size=32)
    print('y_pred:', prob[0])
    
if __name__ == "__main__":
    
    n_folds = 5
    X, y, Xt, yt = load_training_data()
    
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    X = X.astype('float32')
    X /= 255
        
    skf = StratifiedKFold(y.flat, n_folds=n_folds, shuffle=True)

    for train_index, test_index in skf:
            model = create_model()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_and_evaluate_model(model, X_train, y_train, X_test, y_test, False)
    
    #Xt, yt = load_test_data()
    
    X = Xt.reshape(Xt.shape[0], 1, img_rows, img_cols)
    Xt = Xt.astype('float32')
    Xt /= 255
    Yt = np_utils.to_categorical(yt, nb_classes)
    print('will now start final testing')
    train_and_evaluate_model(model, X_train, y_train, Xt, yt, True)
