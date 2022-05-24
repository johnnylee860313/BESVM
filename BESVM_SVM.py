from svm import LinearSVM, KernelSVM
from util.generate_data import generate_data, plot_data_separator, \
    train_test_split
from util.metric import accuracy
import logging
import numpy as np
import random as rd
import pandas as pd

dataset = pd.read_csv("trainingset.csv",names = ['x' + str(i) for i in range(1,11)] + ['y'],sep = "\t")
dataset_testing = pd.read_csv("TestingSet.csv",names = ['x' + str(i) for i in range(1,11)] + ['y'],sep = "\t")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_testing = dataset_testing.iloc[:, :-1].values
y_tesing = dataset_testing.iloc[:, -1].values

tempt_train = 0
tempt_test = 0
for i in range(100):
    X_train, y_train, X_test, y_test = train_test_split(X, y, prop=0.25)
    clf1 = LinearSVM()
    clf1.fit(X=X_train, y=y_train, soft=True)
    acc_train =  accuracy(clf1, X=X_train, y=y_train)
    acc_test =  accuracy(clf1, X=X_test, y=y_test)
    tempt_train = tempt_train + acc_train
    tempt_test = tempt_test + acc_test

logging.info('Original Accuracy (on training) = {}'.format(tempt_train/100))
logging.info('Original Accuracy (on validate) = {}'.format(tempt_test/100))

#random generate numbers from -0.1 to 0.1
#add in to the data
tempt_train = 0
tempt_test = 0
for i in range(100):
    noise = []
    for x in range(len(X)) :
        for i in range(10):
         r = rd.uniform(-0.1,0.1)
         noise.append(r)
    noise = np.array(noise)
    noise = np.reshape(noise,(1000,10))
    X = np.array(X)
    X += noise
    X_train, y_train, X_test, y_test = train_test_split(X, y, prop=0.25)
    clf2 = LinearSVM()
    clf2.fit(X=X_train, y=y_train, soft=True)
    acc_train = accuracy(clf2, X=X_train, y=y_train)
    acc_test =  accuracy(clf2, X=X_test, y=y_test)
    tempt_train = tempt_train + acc_train
    tempt_test = tempt_test + acc_test

print("===================")
logging.info('BESVM Accuracy (on training) = {}'.format(tempt_train/100))
logging.info('BESVM Accuracy (on validate) = {}'.format(tempt_test/100))
print("===================")
acc_testing_Original = accuracy(clf1, X=x_testing, y=y_tesing)
acc_testing_BESVM = accuracy(clf2, X=x_testing, y=y_tesing)
logging.info(' Original Accuracy (on testing) = {}'.format(acc_testing_Original))
logging.info(' BESVM Accuracy (on testing) = {}'.format(acc_testing_BESVM))
