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
from metrics_file import MyMetrics
import csv

batch_size = 16
nb_classes = 2
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 3, 11
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def load_training_data():
    with open("/home/vivek/anaconda2/split_data/breast_tumor_train.csv") as f:    
        data = np.array(list(csv.reader(f, delimiter=",")))
    
    train = []
    #mega = []
    labels = []

    i=0
    for i in data[1:]:
        a = i[:-1]
        a = a[:,np.newaxis]
        a = np.reshape(a, (3,11))
        train.append(a)
        
        b = i[-1]
        labels.append(b)
            
    
    X = np.array(train[:], dtype=np.float)
    y = np.array(labels[:], dtype=np.int)

    return X, y

def load_test_data():
    
    with open("/home/vivek/anaconda2/split_data/breast_tumor_test.csv") as f:    
        data = np.array(list(csv.reader(f, delimiter=",")))
    
    train = []
    labels = []

    i=0
    for i in data[1:]:
        a = i[:-1]
        a = a[:,np.newaxis]
        a = np.reshape(a, (3,11))
        train.append(a)
        
        b = i[-1]
        labels.append(b)
            
    Xt = np.array(train[:], dtype=np.float)
    yt = np.array(labels[:], dtype=np.int)

    return Xt, yt

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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    
    print ('this is shape of y_test', y_test.shape)
    print (type(y_test))
    
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    print ('this is shape of y_train', y_train.shape)
    print (type(y_train))
    
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    
    score = model.evaluate(X_test, y_test)
    print('Test score:', score[0])
    print('Test Accuracy:', score[1])
    
    classes = model.predict_classes(X_test, batch_size=32)
    prob = model.predict_proba(X_test, batch_size=32)
    print('y_pred:', prob[0])
    
     if (cond):
        totalClasses = [0,1]
        metric = MyMetrics(0)
        metric.compute_metrics(X_test, y_test, classes, prob, totalClasses)
        print ('mean acc', metric.mean_accuracy)
        print ('mean f1', metric.mean_f1_score)
        print ('mean roc area', metric.mean_roc_area)
        print ('mean prec', metric.mean_avg_precision)
        print ('mean recall',metrics.recall_score)

if __name__ == "__main__":
    
    n_folds = 5
    X, y = load_training_data()
    
    skf = StratifiedKFold(y.flat, n_folds=n_folds, shuffle=True)

    for train_index, test_index in skf:
            model = create_model()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    
    Xt, yt = load_test_data()
    
    Yt = np_utils.to_categorical(yt, nb_classes)
    
    #print ('this is type of Yt', type(Yt))
    #print ('this is shape of yt', Yt.shape)
    #print (Yt)
    
    train_and_evaluate_model(model, X_train, y_train, Xt, yt)
