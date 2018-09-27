# Splitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

from time import time
import os
import gc
import bottleneck as bn

# Define Hyperparameters
T = 30

def train_test_split(x, y):
    '''
    T = 30,
    feature number = 21
    train:test = 35000-T:5000
    此时的情况是：T = 30，
                  40000 * 1082（21个features）的input x，被拉长为4.328e7 * 21 的数列。
                  39970 * 1082（1个output return）的output，被拉长为4.3248e7 * 21的数列。
    举简单的例子：用T = 3的情况，此时前面三个没有数字
 X：| | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
 P：| | | | | | | |
 Y:       | | | | | | | |
    但是后面三个predicted y其实也没有（因为没有相对应的price做预测）
    既然是相对收益率，那我认为应当截去X的头部30（行）（因其无法产生任何有效的收益率数值）
    train_test分裂暂时使用35000-30:5000
        x_numpy = x_numpy[,:0:1:,]
    '''
    stock_val = 1082
    T = 30
    feature_num = 21
    train_num = 34970
    test_num = 5000

    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(-1,1)
    # Cut first 30 rows of x
    # 30 * 1082 = 32460
    x = x[:,32460:,:]
    '''

    # transfer nan to 0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    '''
    print(np.shape(x))
    print(y.shape)

    # cutting first
    train_X = np.squeeze(x[:,:37837540,:]).T    # 3-D to 2-D
    train_y = y[:37837540,:]
    test_X = np.squeeze(x[:,37837540:,:]).T
    test_y = y[37837540:,:]

    train_X= np.nan_to_num(train_X)
    train_y= np.nan_to_num(train_y)
    test_X= np.nan_to_num(test_X)
    test_y= np.nan_to_num(test_y)

    print("The shape of training set's X is,",train_X.shape)
    print("The shape of training set's y is,",train_y.shape)
    print("The shape of testing set's X is,",test_X.shape)
    print("The shape of testing set's y is,",test_y.shape)


    return train_X, train_y, test_X, test_y


def rolling_splitter(x, y, cut = 8, delete = None):
    '''
    architecture: split the dataset into 8 parts, use prior dataset to predict the next one.

    Return a list of x and a list of y, should contain
    '''
    # print(x.shape)
    x = np.squeeze(x)
    if delete != None:
        x = np.delete(x, delete, axis = 1)
    # x = x.T
    # define hyper-parameter and return paras:
    x_ = []
    y_ = []

    length = 40000 * 1082 / cut
    x = np.asarray(x)
    # x = np.delete(x, [3, 8, 16], axis = 0)
    y = np.asarray(y)
    y = y.reshape(-1,1)
    '''
    # Cut first 30 rows of x
    # 30 * 1082 = 32460
    x = x[32460:,:]


    # transfer nan to

     0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    '''
    print(np.shape(x))
    print(y.shape)

    # cutting first
    for i in range(cut):
        if i != cut-1:
            x_.append(x[int(i*length):int(i*length + length),:])
            y_.append(y[int(i*length):int(i*length + length),:])
        elif i == cut-1:
            x_.append(x[int(i*length):int(i*length + length - 5394330 + 5361870),:])
            y_.append(y[int(i*length):int(i*length + length),:])

    # deleting data if it have nan value inside of it.
    for i in range(cut):
        print('in %d th iteration:'%i)
    # deleting rows form train dataset
        train_index = []
        train_X = x_[i]
        train_y = y_[i]
        print(train_y.shape)
        for j in range(train_y.shape[0]):

            if np.isnan(train_y[j]):
                train_index.append(j)
        train_X = np.delete(train_X, train_index, axis = 0)
        train_y = np.delete(train_y, train_index, axis = 0)
        x_[i] = train_X
        y_[i] = train_y
        print(train_y.shape)
        print(train_X.shape)
        assert np.shape(train_X)[0] == np.shape(train_y)[0]

    print("The shape of total X is,",np.shape(x_))
    print("The shape of total y is,",np.shape(y_))
    '''
    我们的最终目标是返回一个40000*1082的matrix
    我们需要一组完整的x， 一组去掉nan的x， 一组去掉nan的y。
    去掉nan的x/y用来计算mse，rmse，IC index
    不去掉的用来计算整体，方法是如果是第一个model就计算10000*1082，其他的计算5000*1082
    '''

    return x_, y_, x


def twod_splitter(x, y):
    '''
    2-D splitter
    T = 30
    training = 前19500
    testing = 后19970
    先截去后30个x
    截去前500个y
    再分
    train_x: 0到1082*19500
    train_y: 1082*500到1082*20000
    test_x: 1082*19500到1082*39470
    test_y: 1082*20000到1082*39970

    '''
    # define hyper-parameter:


    x = np.asarray(x)
    # x = np.delete(x, [3, 8, 16], axis = 0)
    y = np.asarray(y)
    x = x.reshape(17,-1).T
    y = y.reshape(-1,1)
    '''
    # Cut first 30 rows of x
    # 30 * 1082 = 32460
    x = x[32460:,:]


    # transfer nan to 0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    '''
    print(np.shape(x))
    print(y.shape)

    # cutting first
    train_X = x[:21099000,:]    # 3-D to 2-D
    train_y = y[541000:21640000,:]
    test_X = x[21099000:42706540,:]
    test_y = y[21640000:43247540,:]


    # deleting data if it have nan value inside of it.

    # deleting rows form train dataset
    train_index = []
    for i in range(train_y.shape[0]):
        if np.isnan(train_y[i]):
            train_index.append(i)
    train_X = np.delete(train_X, train_index, axis = 0)
    train_y = np.delete(train_y, train_index, axis = 0)

    # deleting rows form test dataset
    test_index = []
    for i in range(test_y.shape[0]):
        if np.isnan(test_y[i]):
            test_index.append(i)
    test_X = np.delete(test_X, test_index, axis = 0)
    test_y = np.delete(test_y, test_index, axis = 0)


    '''
    train_X= np.nan_to_num(train_X)
    train_y= np.nan_to_num(train_y)
    test_X= np.nan_to_num(test_X)
    test_y= np.nan_to_num(test_y)
    '''
    print("The shape of training set's X is,",train_X.shape)
    print("The shape of training set's y is,",train_y.shape)
    print("The shape of testing set's X is,",test_X.shape)
    print("The shape of testing set's y is,",test_y.shape)


    return train_X, train_y, test_X, test_y


def timing_splitter(x, y):
    '''
    截取出一个3D的X数组
    '''
    # define hyper-parameter:


    x = np.asarray(x)
    # x = np.delete(x, [3, 8, 16], axis = 0)
    y = np.asarray(y)[:,np.newaxis]

    '''
    # Cut first 30 rows of x
    # 30 * 1082 = 32460
    x = x[32460:,:]


    # transfer nan to 0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    '''
    print(np.shape(x))
    print(y.shape)

    # cutting first
    train_X = x[:19500,:,:]    # 3-D to 2-D
    train_y = y[541000:21640000,:]
    test_X = x[19500:39470,:,:]
    test_y = y[21640000:43247540,:]


    # deleting data if it have nan value inside of it.

    # deleting rows form train dataset
    train_index = []
    for i in range(train_y.shape[0]):
        if np.isnan(train_y[i]):
            train_index.append(i)
    # train_X = np.delete(train_X, train_index, axis = 0)
    # train_y = np.delete(train_y, train_index, axis = 0)
    train_y = train_y.reshape(19500,-1)
    train_y = np.nan_to_num(train_y)

    # deleting rows form test dataset
    test_index = []
    for i in range(test_y.shape[0]):
        if np.isnan(test_y[i]):
            test_index.append(i)
    # test_X = np.delete(test_X, test_index, axis = 0)
    # test_y = np.delete(test_y, test_index, axis = 0)
    test_y = test_y.reshape(19970,-1)
    test_y = np.nan_to_num(test_y)

    '''
    train_X= np.nan_to_num(train_X)
    train_y= np.nan_to_num(train_y)
    test_X= np.nan_to_num(test_X)
    test_y= np.nan_to_num(test_y)
    '''
    print("The shape of training set's X is,",train_X.shape)
    print("The shape of training set's y is,",train_y.shape)
    print("The shape of testing set's X is,",test_X.shape)
    print("The shape of testing set's y is,",test_y.shape)


    return train_X, train_y, test_X, test_y
