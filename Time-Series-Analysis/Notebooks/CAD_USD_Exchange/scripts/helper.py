#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:54:37 2020

@author: jnsofini
"""

# Matplotlib  for data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

# Seaborn for data visualization
import seaborn as sns
# Set font scale and style
sns.set(font_scale=2)
sns.set_style('ticks')
plt.style.use('seaborn-white')
mpl.rcParams['font.family'] = 'serif'

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#================================================================


def plotTimeseries(timeseries, color='blue', linestyle=None, marker=None, title=''):
    """
    Makes a plot of the timeseries pandas dataframe input
    
    Inputs
    ------
    timeseries: Timeseries data input as a dataframe
    color: optional with blue as default
    linestyle is optional
    """

    # Set font size and background color
    sns.set(font_scale=2)
    plt.style.use('ggplot')

    ax = timeseries.plot(marker=marker, color=color,
                    linestyle=linestyle, figsize=(15, 6))
    ax.legend(['CAD-USD'])
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('CAD to USD exchange', fontsize=20)
    plt.title(title, fontsize=20)
    plt.grid(True)

#-----------------------------------------------------------------
    
    
def plotStationarityTests(timeseries, title='', nlags=None):
    """This function makes plots 
    
    timeseries is the time series 
    Original time series
    Rolling mean
    Standard deviations
    ACF and partial ACF
    
    Inputs
    ------
    timeseries: dataframe timeseries to make test on and plot
    
    optional parametes title and nlags
    

    """

    # Set font size and background color
    # Set font size and background color
    sns.set(font_scale = 1.5)
    plt.style.use('ggplot')
    
    gridsize = (2, 2)
    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 1))
    
    # Rolling statistic
    rolling_mean = timeseries.rolling(window = 24).mean()
    rolling_std = timeseries.rolling(window = 24).std()
    
    # Plot original time series and rolling mean & std
    timeseries.plot(color = 'r', ax = ax1)
   
    rolling_mean.plot(color = 'b', ax = ax1)
    #ax1.legend(['Original-----'])
    rolling_std.plot(color = 'g', ax = ax1, linestyle='--')
    ax1.legend(['Original', 'Rolling Mean','Rolling STD'])
    ax1.set_ylabel('CAD to USD Exchange', fontsize = 20)

    
    # Plot ACF
    plot_acf(timeseries, lags = nlags, ax = ax2)
    ax2.set_xlabel('Lag', fontsize = 20)
    ax2.grid(True)

    
    # Plot PACF
    plot_pacf(timeseries, lags = nlags, ax = ax3, label=False)
    ax3.set_xlabel('Lag', fontsize = 20)
    ax3.grid(True)
    #ax3.legend(['Original'])
    plt.tight_layout()
    plt.show()
        
    # Perform Dickey-Fuller test
    adf_results = adfuller(timeseries.values.reshape(-1)) 
    print('Test statistic: %0.2f'%(adf_results[0]))
    print('p-value: %0.2f'%(adf_results[1]))
    for key, value in adf_results[4].items():
        print('Critial Values (%s): %0.2f'%(key,value))


def mae(y_test, y_pred):
    """Mean absolute error."""

    mae = np.abs(y_test - y_pred).mean()
    return mae


def rmse(y_test, y_pred):
    """Root mean squared error."""

    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse


def r_squared(y_test, y_pred):
    """r squared (coefficient of determination)."""

    mse = np.mean((y_test - y_pred)**2)  # mean squared error
    var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
    r_squared = 1 - mse / var

    return r_squared


def plotDiagnosticFocast(data, y_true, y_pred, nlags):
    """
    Diagnostic Plot of Residuals for the Out-of-Sample Forecast
    """
    sns.set(font_scale=1.5)
    plt.style.use('ggplot')

    gridsize = (1, 2)
    fig = plt.figure(figsize=(20, 6))
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax2 = plt.subplot2grid(gridsize, (0, 1))
#     ax3 = plt.subplot2grid(gridsize, (1, 0))
#     ax4 = plt.subplot2grid(gridsize, (1, 1))

    residual = y_pred - y_true  # compute the residual

    ax1.scatter(y_pred, residual)
    ax1.set_xlim([min(y_true) - 0.02, max(y_true) + 0.02])
    ax1.axhline(y=0, lw=2, color='k')
    ax1.set(xlabel='Predicted value',  ylabel='Residual', title='Residual plot', fontsize=20)

    ax2.scatter(y_pred, y_true)
    ax2.plot([min(y_true) - 0.02, max(y_true) + 0.02],
             [min(y_true) - 0.02, max(y_true) + 0.02],
             color='k')
    ax2.set_xlim([min(y_true) - 0.02, max(y_true) + 0.02])
    ax2.set_ylim([min(y_true) - 0.02, max(y_true) + 0.02])
    ax2.set_xlabel('Predicted value', fontsize=20)
    ax2.set_ylabel('Actual value', fontsize=20)
    ax2.set_title('Residual plot', fontsize=20)


    

#------------------------------------------------------------------------------------------------------------


def standardizer(X_train, X_test):
    # Instantiate the class
    scaler = StandardScaler()

    # Fit transform the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Only transform the test set
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
#------------------------------------------------------------------------------------------------------------


def mae(y_test, y_pred):
    """Mean absolute error."""
    mae = np.mean(np.abs((y_test - y_pred)))
    return mae

#------------------------------------------------------------------------------------------------------------

def rmse(y_test, y_pred):
    """Root mean squared error."""
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse

#------------------------------------------------------------------------------------------------------------

def r_squared(y_test, y_pred):
    """r-squared (coefficient of determination)."""
    mse = np.mean((y_test - y_pred)**2)  # mean squared error
    var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
    r_squared = 1 - mse / var

    return r_squared

#------------------------------------------------------------------------------------------------------------


def Test_prediction(model, n_training_samples, n_training_label,
                    n_test_samples, n_test_label):
    """Test prediction function"""
    model.fit(n_training_samples, n_training_label)
    test_pred = model.predict(n_test_samples)
    return test_pred

#------------------------------------------------------------------------------------------------------------
def diagnostic_plot(y_pred, y_true):
    """Diagnostic plot"""
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    residual = y_pred - y_true
    r2 = round(r_squared(y_true, y_pred), 3)
    rm = round(rmse(y_true, y_pred), 3)

    ax[0].scatter(y_pred, residual, color='b')
    ax[0].set_xlim([0.7, 1])
    ax[0].hlines(y=0, xmin=-0.1, xmax=1, lw=2, color='r')
    ax[0].set_xlabel('Predicted values')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residual plot')

    ax[1].scatter(y_pred, y_true, color='b')
    ax[1].plot([0.7, 1], [0.7, 1], color='r')
    ax[1].set_xlim([0.7, 1])
    ax[1].set_ylim([0.7, 1])
    ax[1].text(.75, .98, r'$R^2 = {},~ RMSE = {}$'.format(
        str(r2), str(rm)), fontsize=20)
    ax[1].set_xlabel('Predicted values')
    ax[1].set_ylabel('Actual values')
    ax[1].set_title('Residual plot')

#------------------------------------------------------------------------------------------------------------

def plotTrainDiagnostic(fitted, nlags):
    
    sns.set(font_scale = 1.5)
    plt.style.use('ggplot')
    
    gridsize = (2, 2)
    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax2 = plt.subplot2grid(gridsize, (0, 1))
    ax3 = plt.subplot2grid(gridsize, (1, 0))
    ax4 = plt.subplot2grid(gridsize, (1, 1))
    
    residual = model_fit.resid # compute the residual from ARIMA
    
    ax1.scatter(fitted[1:], residual)
    ax1.axhline(y=0, lw=2, color='k')
    ax1.set_xlabel('Fitted value', fontsize = 20)
    ax1.set_ylabel('Residual', fontsize = 20)
    ax1.set_title('Residual plot', fontsize = 20)
    
    from statsmodels.graphics.gofplots import qqplot
    qqplot(model_fit.resid, line='s', ax = ax2, color = 'b')
    ax2.set_title('Normal Q-Q', fontsize = 20)
    ax2.set_xlabel('Sample Quantiles', fontsize = 20)
    ax2.set_ylabel('Theoretical Quantiles', fontsize = 20)
    
    plot_acf(residual, lags = nlags, ax = ax3)
    ax3.set_xlabel('Lag', fontsize = 20)
    ax3.set_ylabel('ACF', fontsize = 20)
    ax3.set_title('Autocorrelation', fontsize = 20)
    ax3.set_xticks([1,5,10,15,20,25,30,35,40,45])
    ax3.set_xlim([.1,42])
    ax3.set_ylim([-0.2,0.2])
    
    plot_pacf(residual, lags = nlags,ax = ax4)
    ax4.set_xlabel('Lag', fontsize = 20)
    ax4.set_ylabel('PACF', fontsize = 20)
    ax4.set_title('Partial Autocorrelation', fontsize = 20)
    ax4.set_xticks([1,5,10,15,20,25,30,35,40,45])
    ax4.set_xlim([.1,42])
    ax4.set_ylim([-0.2,0.2])
    
#-----------------------------------------------------------------------------------
    
def testForecast(train, test):
    
    X_pred = []
    X_true = []
    listcom = [x for x in train] # List comprehension of training data
    
    for i in range(len(test)):
        
        model = ARIMA(listcom, order=(2, 1, 1))  
        model_fit = model.fit() 
        pred = model_fit.forecast()[0]
        X_pred.append(float(pred))
        X_true.append(test[i])
        listcom.append(test[i])
        
    df = pd.DataFrame({'Actual':np.exp(X_true), 'Forecast':np.exp(X_pred)}, 
                      index = test.index)
    
    return df