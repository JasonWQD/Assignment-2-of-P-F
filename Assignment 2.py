#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date:
    Created on Thu Jan 25 09:30:11 2024

Purpose:
    Assignment 2 of Predication & Forecasting
    
Author:
    Thao Le
    Yuanyuan Su
    Jason Wang
"""

###############################################################
### import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

###########################################################
### fData()
def fData(lNames):
    
    vBike = pd.read_excel(lNames[0], index_col = 0)['Bicycle'].values
    vGas1 = pd.read_excel(lNames[1], index_col = 0)['Gasoline'].values
    vGas2 = pd.read_excel(lNames[2], index_col = 0)['Gasoline'].values
    vUmbrella = pd.read_excel(lNames[3])['Umbrella Sales'].values
    mDataAssignment1 = pd.read_excel(lNames[4]).values
    vSun = pd.read_csv(lNames[5], sep = ';', header = None).iloc[-84:, :][3].values

    return vBike, vGas1, vGas2, vUmbrella, mDataAssignment1, vSun

###########################################################
### fStationarity()
def fStationarity(vData):

    print('Augmented Dicky Fuller Test:')
    result = adfuller(vData)
    print('Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("Conclusion:", "\u001b[32mStationary\u001b[0m")
    else:
        print("Conclusion:", "\x1b[31mNon-stationary\x1b[0m")
    
    return

###########################################################
### fACF_PACF()
def fACF_PACF(vData):
    
    plt.figure(dpi = 300)
    sns.lineplot(np.array(range(1, len(vData) + 1)), vData, marker='o')
    plt.tight_layout()
    plt.show()
    
    f, ax = plt.subplots(nrows = 2, ncols = 1, dpi = 300)
    plot_acf(vData, lags = 5, ax = ax[0])
    plot_pacf(vData,lags = 5, ax = ax[1], method = 'ols')
    plt.tight_layout()
    plt.show()

    return 

###########################################################
### fEstimation()
def fEstimation(vData):
    
    vTrain = vData[: -6]
    mod = AutoReg(vTrain, lags = 1, old_names = False)
    res = mod.fit()
    print(res.summary())
    pred = res.predict(start = len(vTrain), end = len(vData) - 1, dynamic = False)
    
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.lineplot(x=sample.timestamp[train_len:num_samples], y=sample.t[train_len:num_samples], marker='o', label='test', color='grey')
    sns.lineplot(x=sample.timestamp[:train_len], y=train, marker='o', label='train')
    sns.lineplot(x=sample.timestamp[train_len:num_samples], y=pred, marker='o', label='pred')
    ax.set_xlim([sample.timestamp.iloc[0], sample.timestamp.iloc[-1]])
    plt.tight_layout()
    plt.show()
    
    
    
    
    return 

###############################################################
### main 
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx', 'Sunspot.csv']
    vBike, vGas1, vGas2, vUmbrella, mDataAssignment1, vSun = fData(lNames)
    
    
    


###########################################################
### start main
if __name__ == "__main__":
    main()