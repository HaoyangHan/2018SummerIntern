# This is the evaluator to for rating factor's performance.
# Version 1.1


'''
Several Explainations:

    Input X: your factor
    Input Y: Price(shape: 40320*1829)

    quantile: unless specified, it should be 10

    Information Correlation = nanmean(IC_)
    Information Ratio = nanmean(IC_)/nanstd(IC_)

    Sharpe Ratio = nanmean(LS_)/nanstd(LS_)
'''

'''
several possible package needed to run the code.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")


import bottleneck as bn
import time


def inf_transfer(x):
    # where_are_nan = np.isnan(x)
    where_are_inf = np.isinf(x)
    # x[where_are_nan] = 0
    x[where_are_inf] = np.nan
    return x

def factor_analysis(X, Y, quantile=10, window=500, minCnt=250):
    '''
    for this factor_analysis program, similar to alpha analysis, we should get 4 outputs:
    IC
    IR
    Sharpe Ratio
    Return Rate Plotting Chart.
    '''
    IC_, LS_ = getStatsTS(X, Y, quantile=10, window=500, minCnt=250)
    # IR_ = np.sqrt(IC_)
    print('IC_ should be:', IC_)
    print('LS_ should be:', LS_)
    IC = np.nanmean(IC_)
    IR = np.nanmean(IC_)/np.nanstd(IC_)

    Sharpe = np.nanmean(LS_)/np.nanstd(LS_)*np.sqrt(240*250)


    print('Statistical factors evaluation start:')
    print("IC should be %f"%IC)
    print("IR should be %f"%IR)
    print("Sharpe Should be %f"%Sharpe)

    # Y_i_ 40320 * 1
    # Y_i  40320/250 * 1

    Y_10_ = np.zeros(Y.shape)
    Y_20_ = np.zeros(Y.shape)
    Y_30_ = np.zeros(Y.shape)

    for i in range(len(Y)-10):
        for j in range(Y.shape[1]):
            Y_10_[i,j] = (Y[i+10,j]-Y[i,j])/Y[i,j]/10
    for i in range(len(Y)-20):
        for j in range(Y.shape[1]):
            Y_20_[i,j] = (Y[i+20,j]-Y[i,j])/Y[i,j]/20
    for i in range(len(Y)-30):
        for j in range(Y.shape[1]):
            Y_30_[i,j] = (Y[i+30,j]-Y[i,j])/Y[i,j]/30


    Y_10 = np.zeros((int(Y.shape[0]/240-1),1))
    Y_20 = np.zeros((int(Y.shape[0]/240-1),1))
    Y_30 = np.zeros((int(Y.shape[0]/240-1),1))


    for i in range(int(len(Y)/240 -1)):
        for j in range(240):
            Y_10[i] += np.nanmean(inf_transfer(Y_10_[240*i+240+j,:]))
    for i in range(int(len(Y)/240 -1)):
        for j in range(240):
            Y_20[i] += np.nanmean(inf_transfer(Y_20_[240*i+240+j,:]))
    for i in range(int(len(Y)/240 -1)):
        for j in range(240):
            Y_30[i] += np.nanmean(inf_transfer(Y_30_[240*i+240+j,:]))


    print('Start plotting.....')
    plt.figure(figsize = (12, 8))
    #plt.subplot(131)
    plt.plot(Y_10, 'b', label = 'ret10')
    plt.plot(Y_20, 'g', label = 'ret20')
    plt.plot(Y_30, 'r', label = 'ret30')
    plt.title("Return Rate comparison between time 10, 20, 30.")
    plt.legend()
    plt.show()

    # plt.subplot(132)
    plt.figure(figsize = (12, 8))
    plt.plot(IC_, label = 'IC')
    plt.title("Information Correlation tendency")
    plt.show()

    # plt.subplot(133)
    plt.figure(figsize = (12, 8))
    plt.plot(IR_, label = 'IR')
    plt.title("Information Ratio tendency")
    plt.show()
    return IC, IR, Sharpe, IC_, LS_

def simple_analysis(X, Y, quantile=10, window=500, minCnt=250):
    '''
    Only calculate IC, IR and Sharpe value without plotting return rate
    return same factor
    '''
    IC_, LS_ = getStatsTS(X, Y, quantile=10, window=500, minCnt=250)
    # IR_ = np.sqrt(IC_)
    print('IC_ should be:', IC_)
    print('LS_ should be:', LS_)
    IC = np.nanmean(IC_)
    IR = np.nanmean(IC_)/np.nanstd(IC_)

    Sharpe = np.nanmean(LS_)/np.nanstd(LS_)*np.sqrt(240*250)


    print('Statistical factors evaluation start:')
    print("IC should be %f"%IC)
    print("IR should be %f"%IR)
    print("Sharpe Should be %f"%Sharpe)
    return IC, IR, Sharpe, IC_, LS_

def getStatsTS(X, Y, quantile=10, window=500, minCnt=250):
    """
    X: Input factor, shape should be 40320*1082
    Y: Existing factor, price
    Calculate the return of 10, 20 ,30 by
    Standardized Return_i = (Price_t+i-Price_t)/Price_t/i
    """
    def calcFwdRet(price, window=30):
        """
        """
        fwd = np.roll(price, -window, axis=0)
        fwd[-window:, :] = np.nan

        return fwd / price - 1


    print('Now Calculating IC and IR matrix, start counting...')
    t0 = time.time()
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y_ = np.zeros(Y.shape)
    for i in range(len(Y)-30):
        for j in range(Y.shape[1]):
            Y_[i,j] = (Y[i+30,j]-Y[i,j])/Y[i,j]/30

    Y = Y_
    if X.shape != Y.shape:
        print(X.shape)
        print(Y.shape)
        raise
    N = len(X)
    IC = np.zeros((N,))

    bottom = 1.0 / quantile
    top = 1 - bottom

    # ts rank
    X = bn.move_rank(X, window=window, min_count=minCnt, axis=0)
    print(np.isnan(X).sum())
    # norm to [0, 1]
    X = 0.5 * (X + 1)

    # get common data
    X = np.where((~np.isnan(X) & (~np.isnan(Y))), X, np.nan)
    Y = np.where((~np.isnan(X) & (~np.isnan(Y))), Y, np.nan)
    # cross-rank Y
    Y_rk = bn.nanrankdata(Y, axis=1)
    Y_rk /= bn.nanmax(Y_rk, axis=1)[:, np.newaxis]

    # ls
    LS = np.nanmean(np.where(X > top, Y, np.nan), axis=1) \
         - np.nanmean(np.where(X < bottom, Y, np.nan), axis=1)

    # Loop
    for ii in range(N):
        IC[ii] = np.corrcoef(X[ii][~np.isnan(X[ii])], Y_rk[ii][~np.isnan(Y_rk[ii])])[0,1]

    t1 = time.time()
    print("total time used for IC and LS matrix calculation is:", (t1-t0))
    return IC, LS

def describe(x):
    '''
    This is a description function to evaluate the characteristic of single stock.
    See the maximum value of each point.
    Here I used this link: https://jingyan.baidu.com/article/6f2f55a18033aeb5b83e6c41.html
    '''
    x_numpy = np.asarray(x)
    x_numpy = x_numpy[:,0:1082]
    x_max = np.amax(x_numpy, axis = 0)
    x_min = np.amin(x_numpy, axis = 0)
    x_nor = (x_numpy - x_numpy.mean())/x_numpy.std()
    x_max_nor = np.amax(x_nor, axis = 0)
    x_min_nor = np.amin(x_nor, axis = 0)

    print("the shape of this input array is:",x_numpy.shape)
    print("the mean of this input array is:",x_numpy.mean())
    print("the var of this input array is:",x_numpy.var())
    print("25th percentile value is:",np.percentile(x_numpy,25))
    print("75th percentile value is:",np.percentile(x_numpy,75))
    print("Plotting the maximum and minimum value line:\n")

    plt.plot(x_max, label = 'Maximum value for each stock')
    plt.plot(x_min, label = 'Minimum value for each stock')
    plt.xlabel('stocks')
    plt.ylabel('factor values')
    plt.legend()
    plt.show()

    print("Plotting the normalized dataset's character:\n")
    plt.plot(x_max_nor, label = 'Normalized maximum value for each stock')
    plt.plot(x_min_nor, label = 'Normalized minimum value for each stock')
    plt.xlabel('stocks')
    plt.ylabel('Normalized factor values')
    plt.legend()
    plt.show()

def matrix_extender(X, shape_zero = 40320, shape_one = 1829):
    '''
    To fit the evaluation code given by Xiaolong, our output matrix must be (40320*1829)
    '''
    print("Input matrix is a(an) %s" %type(X))
    # extend shape 1
    shape = np.shape(X)
    X = np.asarray(X)
    if shape[1] > shape_one:
        print('Error, input matrix have too much columns(shape[1]).')
        raise
    elif shape[0] > shape_zero:
        print('Error, input matrix have too much rows(shape[0].)')
        raise

    # start extending column(shape[1])
    if shape[1] < shape_one:
        delta_shape_one = shape_one - shape[1]
        nans = np.nan * np.zeros((int(shape[0]), int(delta_shape_one)))
        X = np.c_[X, nans]
        assert X.shape[1] == shape_one

    if shape[0] < shape_zero:
        assert X.shape[1] == shape_one
        delta_shape_zero = shape_zero - shape[0]
        nan = np.nan * np.zeros((int(delta_shape_zero), int(shape_one)))
        print(X.shape)
        print(nan.shape)
        X = np.r_[X, nan]
        assert X.shape[0] == shape_zero

    assert X.shape[0] == shape_zero
    assert X.shape[1] == shape_one

    print('Extension done.')

    return X
