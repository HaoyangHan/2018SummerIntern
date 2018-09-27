'''
A test automatic process for fitting machine leanring models(sklearn/xgboost/lightgbm)
Keras and torch should have different strctures and functions.
'''

from mllibrary.importer import dataset_uploader
from mllibrary.splitter import rolling_splitter
from mllibrary.fitter import *
from mllibrary.evaluator import *


# 1. Import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


import time
import os
import gc
import bottleneck as bn
# Define Hyperparameters
T = 30

import warnings
warnings.filterwarnings("ignore")


def overall_uploader():
    x, y = importer.dataset_uploader()
    zeros = np.zeros((500,9))
    x = np.r_[x, zeros]
    x_roll, y_roll, x__ = splitter.rolling_splitter(x, y)

    return x, y, x_roll, y_roll, x__

def overall_fitter(model, x_roll, y_roll , x__):
    y_hat, acc1, acc2, acc3, time_ = fitter.fitting(linear, x_roll = x_roll, y_roll = y_roll, x__ = x__)

    Method.append(str(model))
    MSE.append(acc1)
    RMSE.append(acc2)
    ic.append(acc3)
    Time.append(time_)
    y_hat = evaluator.matrix_extender(y_hat)
    IC, IR, Sharpe, IC_, LS_ = evaluator.simple_analysis(y_hat, price)

    return y_hat, IC, IR, Sharpe
