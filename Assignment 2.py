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
import scipy as sp
from statsmodels.tsa.api import VAR

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
    sns.lineplot(x = np.array(range(1, len(vData) + 1)), y = vData, marker='o')
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
    plt.figure(dpi = 500, figsize = (10, 6))
    sns.lineplot(x = np.array(range(iTrainlen + 1, len(vData) + 1)), y = vData[iTrainlen: iN], marker='o', label = 'test', color = 'grey')
    sns.lineplot(x = np.array(range(1, iTrainlen + 1)), y = vData[: -iPosition], marker = 'o', label = 'train')
    sns.lineplot(x = np.array(range(iTrainlen + 1, iN + 1)), y = mPred[:, 0], marker = 'o', label = 'AR(1)')
    sns.lineplot(x = np.array(range(iTrainlen + 1, iN + 1)), y = mPred[:, 1], marker = 'o', label = 'MA(1)')
    sns.lineplot(x = np.array(range(iTrainlen + 1, iN + 1)), y = mPred[:, 2], marker = 'o', label = 'ARMA(1, 1)')
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

    best_model = pm.auto_arima(vData, start_p = 0, start_q = 0, max_p = 5, 
                  max_q = 5, max_d = 2, max_P = 4, max_Q = 4, 
                  suppress_warnings = True, trace = False, 
                  out_of_sample_size = 6, scoring = 'mse', stepwise = True)    

    return dfPred, dfEva, best_model

###########################################################
### fStandard_Procedure()
def fBox_Jenkins(vData, iPosition):
    
    fStationarity(vData)
    fACF_PACF(vData)
    dfPred, dfEva, best_model = fEstimation(vData, iPosition)
    
    return dfPred, dfEva, best_model


###########################################################
### fFirstDiff()
def fFirstDiff(mDataAssignment1):
    
    mData = mDataAssignment1[:, -3:]
    mDataDiff = np.diff(mData, axis = 0)
    fStationarity(mDataDiff[:, 2])
    plt.plot(mDataDiff)
    
    return mDataDiff

###########################################################
### fVAREsti()
def fVAREsti(mDataAssignment1):
    
    mData = mDataAssignment1[:, -3:]
    mDataDiff = np.diff(mData, axis = 0)
    mTrain = mDataDiff[: -10]
    mPredict = np.zeros((10, mDataDiff.shape[1]))
    for i in range(10):
        model = VAR(mTrain[: 29 + i]).fit(1)
        mPredict[i] = model.forecast(mTrain[28 + i].reshape(-1, 3), steps = 1) + mData[i + 29]
    
    vEvaVar7 = fEvaluation(mData[31: 41, 0], mPredict[:, 0])
    vEvaVar8 = fEvaluation(mData[31: 41, 1], mPredict[:, 1])
    vEvaVar9 = fEvaluation(mData[31: 41, 2], mPredict[:, 2])
    dfEvaIn = pd.DataFrame(np.vstack((vEvaVar7, vEvaVar8, vEvaVar9)), columns = ['ME' , 'MAE', 'MAPE', 'MSE'])
    dfEvaIn.index = ['Var 7', 'Var 8', 'Var 9']
    
    model = VAR(mTrain).fit(1)
    mPredictOut = model.forecast(mTrain[-1].reshape(-1, 3), steps = 10) + mData[-11: -1]
    vEvaVar7Out = fEvaluation(mData[-10: , 0], mPredictOut[:, 0])
    vEvaVar8Out = fEvaluation(mData[-10: , 1], mPredictOut[:, 1])
    vEvaVar9Out = fEvaluation(mData[-10: , 2], mPredictOut[:, 2])
    dfEvaOut = pd.DataFrame(np.vstack((vEvaVar7Out, vEvaVar8Out, vEvaVar9Out)), columns = ['ME' , 'MAE', 'MAPE', 'MSE'])
    dfEvaOut.index = ['Var 7', 'Var 8', 'Var 9']
    
    return dfEvaIn, dfEvaOut


###########################################################
### fVAREsti()
def fAREsti(mDataAssignment1):
    
    mData = mDataAssignment1[:, -3:]
    mDataDiff = np.diff(mData, axis = 0)
    mTrain = mDataDiff[: -10]
    mPredict = np.zeros((10, mDataDiff.shape[1]))
    for i in range(10):
        AR1 = AutoReg(mTrain[: 29 + i, 0], lags = 1, old_names = False).fit()
        AR2 = AutoReg(mTrain[: 29 + i, 1], lags = 1, old_names = False).fit()
        AR3 = AutoReg(mTrain[: 29 + i, 2], lags = 1, old_names = False).fit()
        mPredict[i, 0] = AR1.predict(start = len(mTrain[: 29 + i, 0]), end = len(mTrain[: 29 + i, 0])) + mData[i + 29, 0]
        mPredict[i, 1] = AR2.predict(start = len(mTrain[: 29 + i, 1]), end = len(mTrain[: 29 + i, 1])) + mData[i + 29, 1]
        mPredict[i, 2] = AR3.predict(start = len(mTrain[: 29 + i, 2]), end = len(mTrain[: 29 + i, 2])) + mData[i + 29, 2]

    vEvaVar7 = fEvaluation(mData[31: 41, 0], mPredict[:, 0])
    vEvaVar8 = fEvaluation(mData[31: 41, 1], mPredict[:, 1])
    vEvaVar9 = fEvaluation(mData[31: 41, 2], mPredict[:, 2])
    dfEvaIn = pd.DataFrame(np.vstack((vEvaVar7, vEvaVar8, vEvaVar9)), columns = ['ME' , 'MAE', 'MAPE', 'MSE'])
    dfEvaIn.index = ['Var 7', 'Var 8', 'Var 9']
    
    mPredictOut = np.zeros((10, 3))
    mPredictOut[:, 0] = AR1.predict(start = len(mTrain) + 1, end = len(mDataDiff)) + mData[-11: -1, 0]
    mPredictOut[:, 1] = AR2.predict(start = len(mTrain) + 1, end = len(mDataDiff)) + mData[-11: -1, 1]
    mPredictOut[:, 2] = AR3.predict(start = len(mTrain) + 1, end = len(mDataDiff)) + mData[-11: -1, 2]
    vEvaVar7Out = fEvaluation(mData[-10: , 0], mPredictOut[:, 0])
    vEvaVar8Out = fEvaluation(mData[-10: , 1], mPredictOut[:, 1])
    vEvaVar9Out = fEvaluation(mData[-10: , 2], mPredictOut[:, 2])
    dfEvaOut = pd.DataFrame(np.vstack((vEvaVar7Out, vEvaVar8Out, vEvaVar9Out)), columns = ['ME' , 'MAE', 'MAPE', 'MSE'])
    dfEvaOut.index = ['Var 7', 'Var 8', 'Var 9']
    
    return dfEvaIn, dfEvaOut

###############################################################
### main 
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx', 'Sunspot.csv']
    vBike, vGas1, vGas2, vUmbrella, mDataAssignment1, vSun = fData(lNames)
    
    # iPosition means how many last forecasts you want to use for the performance evaluation
    iPosition = 10
    dfPred, dfEva, best_model = fBox_Jenkins(mDataAssignment1[:, 8], iPosition)
    
    # Var7, Var8, and Var9 are selected
    dfVAREvaIn, dfVAREvaOut = fVAREsti(mDataAssignment1)
    dfAREvaIn, dfAREvaOut = fAREsti(mDataAssignment1)


###########################################################
### start main
if __name__ == "__main__":
    main()