# Gold Price Prediction Using LSTM and Random Forest Regression

## A python based project that predicts the closing price of gold in commodity exchange using fundamental data and macro-economic data. LSTM and Random Forest models are used here.

## Introduction
The aim of this project is to predict the close price of gold using fundamental data and macro-economic data. The features used for prediction are:
* Open, high and low values from the gold dataset in Yahoo Finance (ticker: GC=F)
* Close price of crude oil from Brent Crude Oil index in Yahoo Finance (ticker: BZ=F)
* Close price of Dow Jones Industrial Average index from Yahoo Finance (ticker: ^DJI)
* Close price of Walmart from Yahoo Finance (ticker: WMT)
* Inflation rate represented by the 10-Year Breakeven Inflation Rate from FRED site (symbol: T10YIE)
* US Federal Reserve interest rate from Federal Funds Effective Rate in FRED site (symbol: FEDFUNDS)

The dataset has datapoints acquired from July 2007 to March 2023. LSTM and Random Forest Regression models are used for getting prediction results. The Root Mean Squared Error (RMSE) & Mean Absolute Percentage Error (MAPE) for each of the models, and their combination are recorded. Hyperparameter tuning through Grid Search is performed at the last stage of the application to find the optimum parameters to be used in LSTM model. The Grid Seach is time-consuming.

## How to get started
The code can be run using PyDev plugin for Eclipse. The code contains certain write and read functions to local files, comment it out if you do not want to create those files.
