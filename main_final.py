#!/usr/bin/env python
# coding: utf-

# In[1]:


# Testing the linear regression package we have created
from Lr_package_final import *


tickers = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL']

start = '2015-01-01'
end = datetime.datetime.today()


# instance of the linear regression package

linearpackage = StockLinearRegression(tickers)

# pulling prices from Yahoo Finance for the date range specified
linreg_prices = linearpackage.PullPrices(start, end)

print("Let's see the first few rows of the prices dataframe: ")
print(linreg_prices.head)


# setting a window of 30 days
linreg_window = linearpackage.DefineWindow(30)
print(linreg_window[0])

# calculating variance and covariance
linreg_variance = linearpackage.CalculateVariance()
print(str(linreg_variance[0]))
linreg_covariance = linearpackage.CalculateCovariance()
print(str(linreg_covariance[0][0]))

# calculating the slopes (Beta) of the dataset
linreg_slopes = linearpackage.CalculateSlopes()
print("Slopes are being calculated(Beta): \n" + str(linreg_slopes[0]))

# calculating Intercepts from the slopes of the dataset
linreg_intercepts = linearpackage.CalculateIntercepts()
print("Intercepts for the chosen stocks are as follows: \n" +
      str(linreg_intercepts[0]))

# plotting the linear regression

linearpackage.PlottingWindow(100, "FB")
linearpackage.PlottingWindow(100, "AMZN")
linearpackage.PlottingWindow(100, "AAPL")
linearpackage.PlottingWindow(100, "NFLX")
linearpackage.PlottingWindow(100, "GOOGL")

# get the R squared value
linreg_rsquared = linearpackage.CalculateRsquared(
    150, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_rsquared)

# get the standard error coefficient
linreg_stdcoefferr = linearpackage.CalculateStdcoeff(
    150, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_stdcoefferr)

# get the T Scores

linreg_tscores = linearpackage.CalculateTscores(
    150, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_tscores)

linreg_tscores = linearpackage.CalculateTscores(
    100, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_tscores)

linreg_tscores = linearpackage.CalculateTscores(
    50, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_tscores)

linreg_tscores = linearpackage.CalculateTscores(
    30, ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_tscores)

# get the P Values

linreg_pvalues = linearpackage.CalculatePvalues(
    ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL'])
print(linreg_pvalues)


# In[4]:


#

# In[ ]:
