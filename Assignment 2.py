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

    result = adfuller(vData)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")



###########################################################
### ACF(mLogRet, iLags, mStats, boolAbs) = mAutoCORR
def ACF(mLogRet, iLags, mStats, boolAbs):
    """
    Purpose:
        Calculate an ACF for iLags amount of lags. The boolean input parameter
        decides whether the absolute values of a series must be used.

    Inputs:
        mLogRet         matrix, filled with columns of log returns
        iLags           integer, number of lags 
        mStats          matrix, filled with stats about 3 log return columns
        boolAbs         boolean, True when using absolute log returns and
                        false otherwise
                        
    Return values:
        mAutoCorr       iLags+1 x iK matrix, contains autocorrelations per column
    """
    iN, iK = mLogRet.shape
    mAutoCov = np.zeros((iLags+1, iK))
    mAutoCorr = np.zeros((iLags+1, iK))
    
    if boolAbs == True:
         mLogRet = np.abs(mLogRet)     
    vMean = np.mean(mLogRet, axis=0)
    
    for i in range(iK):
        vDemeaned = mLogRet[:,i]-vMean[i]
        for k in range(iLags+1):
            mAutoCov[k,i] = (1/iN)*(vDemeaned.T[k:iN] @ vDemeaned[:iN-k])
            mAutoCorr[k,i] = mAutoCov[k,i] / mAutoCov[0,i]

    return mAutoCorr, mAutoCov

###########################################################
### PACF(mYfull, iLags, boolAbs)
def PACF(mYfull, iLags, boolAbs):
    """
    Purpose:
        Calculate a PACF for iLags number of lags. The boolean input parameter
        decides whether the absolute values of a series must be used.

    Inputs:
        mYfull          Matrix of time series in different columns
        iLags           integer, number of lags to be used in PACF function             
        boolAbs         boolean, True when using absolute log returns and
                        false otherwise
                        
    Return values:
        mPACF           (iLags + 1) x iK matrix, first value for each series is 0
    """
    (iN,iK) = mYfull.shape
    mPACF = np.zeros((iLags+1,iK))
                
    if boolAbs == True:
        mYfull = np.abs(mYfull)
            
    for i in range(iK):
        for n in range(1,iLags+1):
            (mX, vY) = CreateX_Y(n, 0, mYfull[:,i])
            XtXi = np.linalg.inv(mX.T @ mX)
            vB = XtXi @ mX.T @ vY
            mPACF[n, i] = vB[-1]
            
    return mPACF
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