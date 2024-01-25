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
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

###############################################################
### fData
def fData(sName):
    
    dfSP = yf.download("^GSPC", start = '1996-01-01', end = '2024-12-31')[['Adj Close']]
    dfSP['Return'] = np.log(dfSP['Adj Close'] / dfSP['Adj Close'].shift(1))
    dfSP = dfSP.iloc[1:, ]
    dfRV = pd.read_csv(sName)[['Date', 'Type', 'Volatility']]
    dfRV = dfRV[dfRV['Type'] == 'QMLE-Trade']
    dfRV['Date'] = pd.to_datetime(dfRV['Date'])
    dfRV = dfRV.set_index('Date')
    dfSP['Volatility'] = dfRV['Volatility']
    dfSP = dfSP.dropna(axis = 0)
    
    return dfSP.iloc[:, 1:]

###############################################################
### main 
def main():
    
    sName = 'S&P daily RV.csv'
    dfSP = fData(sName)
    dfSP.to_csv('Return_and_volatility.csv')
    plt.figure(dpi = 300)
    plt.plot(dfSP.index, dfSP['Return'], label = 'Return', color = 'red')
    plt.plot(dfSP.index, dfSP['Volatility'], label = 'RV')
    plt.legend()
    plt.show()


###########################################################
### start main
if __name__ == "__main__":
    main()