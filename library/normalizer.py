# Normalizer
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

def nan_transfer(x):
    '''
    An auxiliary function to change all nan/inf value into 0
    This is dangerous, since nan value may have its real mean, such as suspension.
    '''
    where_are_nan = np.isnan(x)
    where_are_inf = np.isinf(x)
    x[where_are_nan] = 0
    x[where_are_nan] = 0
    return x


def timing_standardization(x, delete = [5, 11, 17]):
    '''
    for every single factor, for every single time point, calculate prior 500 point's maximum and minimum, do standardization.
    '''

    # window size(for standardization)
    window = 500

    print('Attention! you should input the parameter you wanna delete, or system would automatically delete 3 selected rows.')
    print('For this algorithm, every single factor would use around 7 minutes.')

    x = np.asarray(x)
    if delete != None
        x = np.delete(x, delete, axis = 0)
    print('the shape of input matrix is %d, %d, %d' %(x.shape[0],x.shape[1], x.shape[2]))
    print('This matrix have %d nan value' %(np.isnan(x).sum()))


    t0 = time()
    x_bar = np.zeros((x.shape[0], (x.shape[1]-window), x.shape[2]))
    for i in range(x.shape[0]):
        print('%d th factor:'%i)
        for j in range(x.shape[1]-window):
            for k in range(x.shape[2]):
                x_ = x[i, j:j+500, k].reshape((500, 1))
                x_order = np.argsort(x_)
                max_x = x_[x_order[10]]
                min_x = x_[x_order[490]]
                # assert max_x >= min_x
                if max_x - min_x != 0:
                    x_bar[i,j,k] = (x[i,j+500,k] - min_x)/(max_x - min_x)
                else:
                    x_bar[i,j,k] = 0
        gc.collect()

    t1 = time()
    print('Total transfer time is %d' %(t1-t0))
    print(x_bar.shape)
    x_bar = np.asarray(x_bar)
    return x_bar


def stock_standardization(x, delete = [5, 11, 17]):
    '''
    for every single factor, for every single time point, calculate prior 500 point's maximum and minimum, do standardization.
    '''

    # window size(for standardization)
    window = 500

    x = np.asarray(x)
    x = np.delete(x, delete, axis = 0)
    print('the shape of input matrix is %d, %d, %d' %(x.shape[0],x.shape[1], x.shape[2]))
    print('This matrix have %d nan value' %(np.isnan(x).sum()))


    t0 = time.time()
    x_bar = np.zeros((x.shape[0], (x.shape[1]-window), x.shape[2]))
    for i in range(x.shape[0]):
        print('%d th factor:'%i)
        for j in range(x.shape[1]-window):
            for k in range(x.shape[2]):
                x_ = x[i, j:j+500, k].reshape((500, 1))
                x_order = np.argsort(x_)
                max_x = x_[x_order[10]]
                min_x = x_[x_order[490]]
                # assert max_x >= min_x
                if max_x - min_x != 0:
                    x_bar[i,j,k] = (x[i,j+500,k] - min_x)/(max_x - min_x)
                else:
                    x_bar[i,j,k] = 0
        gc.collect()
        '''

        for j in range(6):
            x_ = x[i * 1082: i* 1082 + 1082, j]
            x_order = np.argsort(x_)
            if i == 0:
                print(type(x_order))
                print(x_[x_order[108]]) # here we use 108
                print(x_[x_order[974]]) # here we use 1082 - 108 = 974
            max_x = x_[x_order[54]] #there exist nan value in this np array
            min_x = x_[x_order[1028]]
            if max_x - min_x != 0:
                x_hat = (x_ - min_x)/(max_x - min_x) # this should be a range(0,1) standardization
                x_hat = 2 * x_hat - 1
                x_bar[i * 1082: i* 1082 + 1082, j] = x_hat
            else:
                x_hat = np.zeros(1082)
                x_bar[i * 1082: i* 1082 + 1082, j] = x_hat
        '''

    t1 = time.time()
    print('Total transfer time is %d' %(t1-t0))
    print(x_bar.shape)
    x_bar = np.asarray(x_bar)
    return x_bar


def sharpe(X):
    if np.shape(X)[0] > np.shape(X)[1]:
        X = X.T
    x_hat = X[:,:10000000]    # avoid memory error

    assert np.isnan(x_hat).sum() == 0

    pearson_ = np.corrcoef(x_hat)
    for i in range(len(pearson_)):
        for j in range(len(pearson_)):
            if i != j:
                if abs(pearson_[i,j]) > 0.6:
                    print("the %d, %d parameters are correlated, it's value should be: %f" % (i, j, pearson_[i, j]))

    sns.heatmap(pearson_)
    del x_hat, pearson_
    gc.collect()
