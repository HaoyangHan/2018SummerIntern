# import all needed files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import warnings
warnings.filterwarnings("ignore")

import time
import os
import gc
import bottleneck as bn

# Define Hyperparameters
T = 30

def import_dataset_before(address):
    '''
    read all 2-byte files and then concentrating them together. Follow this link:
    https://blog.csdn.net/brucewong0516/article/details/79062340

    instead of using alpha as y,
    we should use return rate as y.
    '''
    x = []
    y = []
    # import x
    # address = os.listdir(r'../factors_pickingtime')
    # address = os.listdir(address)
    u = 1
    factors = r'factors_pickingtime/'
    parent_ = os.path.join(address,factors)
    parent = os.listdir(parent_)
    for path in parent:
        # if path in add:

        abs_path = os.path.join(parent_, path)
        print(path)
        x_i = np.memmap(abs_path,dtype = np.float32, shape = (40000,1829))# ,shape = (204800, 1749))
        x_i = np.asarray(x_i)
        x_i = x_i[:,0:1082]
        x_ = x_i.reshape(-1,1)
        print(np.shape(x_))
        print("%d th factor have %d number of nan factors "%(u, np.isnan(x_).sum()))
        x.append(x_)
        # print(abs_path)
        u += 1
    # import y
    price_ = os.path.join(address, 'quotes/stk_clsadj')
    price = np.memmap(price_,dtype = np.float32, shape = (40000,1829))# ,shape = (204800, 1749))

    y = np.zeros((price.shape[0]-T,price.shape[1]))
    '''
    This is an iteration to create target y function;
    Here we use Return rate minus Average Return Rate.

    First we calculate the mean return rate for every single timeslot;
    Then we calculate the return rate of each stock and return them back to y.

    Here we used np.nanmean() function, in case some values would be nan.
    '''
    t0 = time.time()
    for i in range(40000-T):

        for j in range(1082):
            y[i,j-1] = np.log(price[i+T,j-1]/price[i,j-1])
    t1 = time.time()
    y = y[:,0:1082]
    y_ = y.reshape(-1,1)

    # delete too-much nan value column
    # x = np.delete(x, 5)
    del address, x_i, x_, price, y
    gc.collect()
    print("y function running time should be, ", t1-t0)
    print("the shape of x is:", np.shape(x))
    print("the shape of y is:", y_.shape)
    return x, y_



def return_calculator(address):
    bid_ = os.path.join(address, 'stk_bidadj')
    bid = np.memmap(bid_,dtype = np.float32, shape = (40320,1829))# ,shape = (204800, 1749))

    ask_ = os.path.join(address, 'stk_askadj')
    ask = np.memmap(ask_,dtype = np.float32, shape = (40320,1829))# ,shape = (204800, 1749))



    price = (ask + bid)/2
    y = np.zeros((price.shape[0]-T,price.shape[1]))

    for i in range(40000-T):
        for j in range(1082):
            y[i,j-1] = np.log(price[i+T,j-1]/price[i,j-1])
    y = y[:,0:1082]
    y_ = y.reshape(-1,1)

    return y_

def import_dataset(address, factors = ['TSX[100002_1]', 'TSX[100009_1]', 'TSX[100010_2]', 'TSX[100012_1]', 'TSX[100029_1]', 'TSX[100038_1]', 'TSX[100038_2]', 'TSX[100038_3]', 'TSX[100038_4]']
):
    '''
    read all 2-byte files and then concentrating them together. Follow this link:
    https://blog.csdn.net/brucewong0516/article/details/79062340

    instead of using alpha as y,
    we should use return rate as y.
    '''
    x = []
    y = []
    # import x
    # address = os.listdir(r'../factors_pickingtime')
    # address = os.listdir(address)
    u = 1
    factors = r'factors_pickingtime/'
    parent_ = os.path.join(address,factors)
    parent = os.listdir(parent_)
    for path in parent:
        if path in factors:
            abs_path = os.path.join(parent_, path)
            print(path)
            x_i = np.memmap(abs_path,dtype = np.float32, shape = (40000,1829))# ,shape = (204800, 1749))
            x_i = np.asarray(x_i)
            x_i = x_i[:,0:1082]
            x_ = x_i.reshape(-1,1)
            print(np.shape(x_))
            print("%d th factor have %d number of nan factors "%(u, np.isnan(x_).sum()))
            x.append(x_)
            # print(abs_path)
            u += 1

    # import y
    price_ = os.path.join(address, 'quotes/stk_clsadj')
    price = np.memmap(price_,dtype = np.float32, shape = (40000,1829))# ,shape = (204800, 1749))

    y = np.zeros((price.shape[0]-T,price.shape[1]))
    '''
    This is an iteration to create target y function;
    Here we use Return rate minus Average Return Rate.

    First we calculate the mean return rate for every single timeslot;
    Then we calculate the return rate of each stock and return them back to y.

    Here we used np.nanmean() function, in case some values would be nan.
    '''
    t0 = time.time()
    for i in range(40000-T):

        for j in range(1082):
            y[i,j-1] = np.log(price[i+T,j-1]/price[i,j-1])
    t1 = time.time()
    y = y[:,0:1082]
    y_ = y.reshape(-1,1)

    # delete too-much nan value column
    # x = np.delete(x, 5)
    del address, x_i, x_, price, y
    gc.collect()
    print("y function running time should be, ", t1-t0)
    print("the shape of x is:", np.shape(x))
    print("the shape of y is:", y_.shape)
    return x, y_


def dataset_uploader():
    X = np.loadtxt('../../data/x_.csv')
    Y = np.loadtxt('../../data/y.csv')
    return X, Y
