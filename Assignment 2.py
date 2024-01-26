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
from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pm

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
### fPlotPredict()
def fPlotPredict(vData, mPred, iPosition):
    
    iN = len(vData)
    iTrainlen = iN - iPosition
    plt.figure(dpi = 300, figsize = (10, 6))
    sns.lineplot(np.array(range(iTrainlen + 1, len(vData) + 1)), vData[iTrainlen: iN], marker='o', label = 'test', color = 'grey')
    sns.lineplot(np.array(range(1, iTrainlen + 1)), vData[: -iPosition], marker = 'o', label = 'train')
    sns.lineplot(np.array(range(iTrainlen + 1, iN + 1)), mPred[:, 0], marker = 'o', label = 'AR(1)')
    sns.lineplot(np.array(range(iTrainlen + 1, iN + 1)), mPred[:, 1], marker = 'o', label = 'MA(1)')
    sns.lineplot(np.array(range(iTrainlen + 1, iN + 1)), mPred[:, 2], marker = 'o', label = 'ARMA(1, 1)')
    plt.tight_layout()
    plt.show()
    
    return 

###########################################################
### fEvaluation()
def fEvaluation(vYt, vYt_hat):
    
    vUt = vYt - vYt_hat
    dME = round(np.mean(vUt), 2)
    dMAE = round(np.mean(np.abs(vUt)), 2)
    dMAPE = round(100 * np.mean(np.divide(np.abs(vUt), np.abs(vYt))), 2)
    dMSE = round(np.mean(vUt ** 2), 2)
    
    return dME, dMAE, dMAPE, dMSE

###########################################################
### fEstimation()
def fEstimation(vData, iPosition):
    
    mPred = np.zeros((iPosition, 3))
    for i in range(iPosition):
        vTrain = vData[: i - iPosition]
        AR = AutoReg(vTrain, lags = 1, old_names = False).fit()
        MA = ARIMA(vTrain, order = (0, 0, 1)).fit()
        ARMA = ARIMA(vTrain, order = (1, 0, 1)).fit()
        mPred[i, 0] = AR.predict(start = len(vTrain), end = len(vTrain))
        mPred[i, 1] = MA.predict(start = len(vTrain), end = len(vTrain))
        mPred[i, 2] = ARMA.predict(start = len(vTrain), end = len(vTrain))
    
    dfPred = pd.DataFrame(mPred, columns = ['AR(1)', 'MA(1)', 'ARMA(1, 1)'])
    fPlotPredict(vData, mPred, iPosition)
    
    mEva = np.vstack((fEvaluation(vData[-len(mPred): ], mPred[:, 0]), fEvaluation(vData[-len(mPred): ], mPred[:, 1]), fEvaluation(vData[-len(mPred): ], mPred[:, 2])))
    dfEva = pd.DataFrame(mEva, columns = ['ME' , 'MAE', 'MAPE', 'MSE'])
    dfEva.index = ['AR(1)', 'MA(1)', 'ARMA(1, 1)']

    best_model = pm.auto_arima(vData, d = 0, start_p = 0, start_q = 0, max_p = 5, 
                  max_q = 5, max_d = 2, max_P = 4, max_Q = 4, 
                  suppress_warnings = True, trace = True, 
                  out_of_sample_size = 6, scoring = 'mse', stepwise = True)    

    return dfPred, dfEva, best_model

###########################################################
### fStandard_Procedure()
def fBox_Jenkins(vData, iPosition):
    
    fStationarity(vData)
    fACF_PACF(vData)
    dfPred, dfEva, best_model = fEstimation(vData, iPosition)
    
    return dfPred, dfEva, best_model

###############################################################
### main 
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx', 'Sunspot.csv']
    vBike, vGas1, vGas2, vUmbrella, mDataAssignment1, vSun = fData(lNames)
    
    # iPosition means how many last forecasts you want to use for the performance evaluation
    iPosition = 10
    dfPred, dfEva, best_model = fBox_Jenkins(mDataAssignment1[:, 6], iPosition)
    
    


###########################################################
### start main
if __name__ == "__main__":
    main()