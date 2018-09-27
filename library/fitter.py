'''
Super fitter

There should be 3 part inside:

Traditional sklearn machine learning models;
Keras deep learning models;
Pytorch deep learning models;

XGBoost, LibSVM, etc.......

there should be a standardized procedure for fitting for machine learning models.
'''

# Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import time
import os
import gc
import bottleneck as bn

import warnings
warnings.filterwarnings("ignore")

# Accuracy
from sklearn.metrics import accuracy_score as acc_s
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics
'''
Since this is a regression problem instead of classification,
we shouldn't use accuracy_score(compared one element from each other in this case .)
Use MSE and RMSE criteria instead.

'''

def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))

def IC(x1, x2):
    pearson = []
    x1 = np.squeeze(x1)
    x2 = np.squeeze(x2)
    for i in range(int(np.shape(x1)[0]/1082)):
        x1_ = x1[i*1082:i*1082+1082,]
        x2_ = x2[i*1082:i*1082+1082,]
        pearson_ = pearsonr(x1_, x2_)[0]
        pearson.append(pearson_)
    pearson = np.asarray(pearson)
    pearson = np.squeeze(pearson)
    '''
    print(np.isnan(pearson).sum())
    print(pearson.shape)
    print(pearson.mean())
    print(pearson.std())
    '''
    temp = pearson.mean()/pearson.std()
    return temp

def nan_transfor(x):
    where_are_nan = np.isnan(x)
    where_are_inf = np.isinf(x)
    x[where_are_nan] = 0
    x[where_are_nan] = 0
    return x

def predictor(func):
    y_hat = func.predict(pred_x)
    y_hat = y_hat.reshape(40000,-1)
    return y_hat

def fitting(func, x_roll, y_roll, x__):
    '''
    diivding the dataset into 8 parts. Use former part as training dataset and latter part as testint dataset.
    '''
    t0 = time.time()
    print('model is %s' %func)
    # define several returnable parameters
    acc_1 = []    # mse
    acc_2 = []    # rmse
    acc_3 = []    # IC
    length = 40000 *1082 / 8

    assert np.shape(x_roll)[0] == np.shape(y_roll)[0] == 8

    for i in range(np.shape(x_roll)[0]-1):

        train_X = np.asarray(x_roll[i])
        test_X = np.asarray(x_roll[i+1])
        train_y = np.asarray(y_roll[i])
        test_y = np.asarray(y_roll[i+1])
        print('%d th iteration'%i)
        print(test_X.shape[1])
        print(test_y.shape[0])
        # assert train_X.shape[1] == 17
        assert train_X.shape[0] == train_y.shape[0]
        assert test_X.shape[0] == test_y.shape[0]

        '''
        我们的最终目标是返回一个40000*1082的matrix
        我们需要一组完整的x， 一组去掉nan的x， 一组去掉nan的y。
        去掉nan的x/y用来计算mse，rmse，IC index
        不去掉的用来计算整体，方法是如果是第一个model就计算10000*1082，其他的计算5000*1082
        '''
        func.fit(train_X, train_y)
        pred_y = func.predict(test_X)
        if pred_y.ndim == 1:
            pred_y = pred_y[:,np.newaxis]
        assert np.shape(pred_y)[0] == np.shape(test_y)[0]
        acc_1.append(mse(pred_y, test_y))
        acc_2.append(rmse(pred_y, test_y))
        acc_3.append(IC(pred_y, test_y))

        if i == 0:
            output = func.predict(x__[:int(2*length),:])
            if output.ndim == 1:
                output = output[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output)[0], np.shape(output)[1]))
        if i != 0:
            output_ = func.predict(x__[int(length*(i+1)):int((i+2)*length),:])
            if output_.ndim == 1:
                output_ = output_[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output_)[0], np.shape(output_)[1]))
            output = np.concatenate((output, output_))

    accuracy_1 = np.mean(acc_1)
    accuracy_2 = np.mean(acc_2)
    accuracy_3 = np.mean(acc_3)
    output = output.reshape(40000,-1)

    print('average value of mse is %f' %(accuracy_1))
    print('average value of rmse is %f' %(accuracy_2))
    print('average value of IC is %f' %(accuracy_3))

    t1 = time.time()
    print('Total time used for fitting model %s is: %f'%(func,(t1-t0)))

    return output, accuracy_1, accuracy_2, accuracy_3, (t1-t0)

def fitting_deep(func, x_roll, y_roll, x__, **kwargs):
    for key in kwargs:
        print(key)
        print(type(key))
    '''
    diivding the dataset into 8 parts. Use former part as training dataset and latter part as testint dataset.
    '''
    t0 = time.time()
    print('model is %s' %func)
    # define several returnable parameters
    acc_1 = []    # mse
    acc_2 = []    # rmse
    acc_3 = []    # IC
    length = 40000 *1082 / 8

    assert np.shape(x_roll)[0] == np.shape(y_roll)[0] == 8

    for i in range(np.shape(x_roll)[0]-1):

        train_X = np.asarray(x_roll[i])
        test_X = np.asarray(x_roll[i+1])
        train_y = np.asarray(y_roll[i])
        test_y = np.asarray(y_roll[i+1])
        print('%d th iteration'%i)
        print(test_X.shape[1])
        print(test_y.shape[0])
        # assert train_X.shape[1] == 17
        assert train_X.shape[0] == train_y.shape[0]
        assert test_X.shape[0] == test_y.shape[0]

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
                func.fit(train_X, train_y, epochs = epochs, batch_size = batch_size)
            else:
                func.fit(train_X, train_y, batch_size = batch_size)
        elif 'epochs' in kwargs:
            epochs = kwargs['epochs']
            func.fit(train_X, train_y, epochs = epochs)
        else:
            func.fit(train_X, train_y)
        pred_y = func.predict(test_X)
        if pred_y.ndim == 1:
            pred_y = pred_y[:,np.newaxis]
        assert np.shape(pred_y)[0] == np.shape(test_y)[0]
        acc_1.append(mse(pred_y, test_y))
        acc_2.append(rmse(pred_y, test_y))
        acc_3.append(IC(pred_y, test_y))

        if i == 0:
            output = func.predict(x__[:int(2*length),:])
            if output.ndim == 1:
                output = output[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output)[0], np.shape(output)[1]))
        if i != 0:
            output_ = func.predict(x__[int(length*(i+1)):int((i+2)*length),:])
            if output_.ndim == 1:
                output_ = output_[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output_)[0], np.shape(output_)[1]))
            output = np.concatenate((output, output_))

    accuracy_1 = np.mean(acc_1)
    accuracy_2 = np.mean(acc_2)
    accuracy_3 = np.mean(acc_3)
    output = output.reshape(40000,-1)

    print('average value of mse is %f' %(accuracy_1))
    print('average value of rmse is %f' %(accuracy_2))
    print('average value of IC is %f' %(accuracy_3))

    t1 = time.time()
    print('Total time used for fitting model %s is: %f'%(func,(t1-t0)))

    return output, accuracy_1, accuracy_2, accuracy_3, (t1-t0)

def fitting_lightgbm(paras, x_roll, y_roll, x__):
    '''
    specifically for lightgbm.
    '''
    t0 = time.time()
    print('model is %s' %func)
    # define several returnable parameters
    acc_1 = []    # mse
    acc_2 = []    # rmse
    acc_3 = []    # IC
    length = 40000 *1082 / 8

    assert np.shape(x_roll)[0] == np.shape(y_roll)[0] == 8

    for i in range(np.shape(x_roll)[0]-1):

        train_X = np.asarray(x_roll[i])
        test_X = np.asarray(x_roll[i+1])
        train_y = np.asarray(y_roll[i])
        test_y = np.asarray(y_roll[i+1])
        print('%d th iteration'%i)
        print(test_X.shape[1])
        print(test_y.shape[0])
        # assert train_X.shape[1] == 17
        assert train_X.shape[0] == train_y.shape[0]
        assert test_X.shape[0] == test_y.shape[0]

        func.fit(train_X, train_y)
        pred_y = func.predict(test_X)
        if pred_y.ndim == 1:
            pred_y = pred_y[:,np.newaxis]
        assert np.shape(pred_y)[0] == np.shape(test_y)[0]
        acc_1.append(mse(pred_y, test_y))
        acc_2.append(rmse(pred_y, test_y))
        acc_3.append(IC(pred_y, test_y))

        if i == 0:
            output = func.predict(x__[:int(2*length),:])
            if output.ndim == 1:
                output = output[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output)[0], np.shape(output)[1]))
        if i != 0:
            output_ = func.predict(x__[int(length*(i+1)):int((i+2)*length),:])
            if output_.ndim == 1:
                output_ = output_[:,np.newaxis]
            print('the %d th iteration. output shape is (%d,%d)' %(i, np.shape(output_)[0], np.shape(output_)[1]))
            output = np.concatenate((output, output_))

    accuracy_1 = np.mean(acc_1)
    accuracy_2 = np.mean(acc_2)
    accuracy_3 = np.mean(acc_3)
    output = output.reshape(40000,-1)

    print('average value of mse is %f' %(accuracy_1))
    print('average value of rmse is %f' %(accuracy_2))
    print('average value of IC is %f' %(accuracy_3))

    t1 = time.time()
    print('Total time used for fitting model %s is: %f'%(func,(t1-t0)))

    return output, accuracy_1, accuracy_2, accuracy_3, (t1-t0)
