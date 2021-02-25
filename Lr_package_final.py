#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import math
import numpy as np
import scipy.stats as stat
import yfinance as yhf
import matplotlib.pyplot as plt
import pandas_datareader
import pandas_datareader.data as web
import datetime

# Class to pull the data from yahoo finance (yfinance)


class StockLinearRegression:
    def __init__(self, tickers):
        # Stock tickers are converted to the data type character
        tickers = ', '.join(tickers)
        self.tickers = tickers
        # Tickers for this project are taken from NASDAQ index.
        # To pull data from yahoo finance NASDAQ index the ticker has been set to IXIC
        # https://in.finance.yahoo.com/quote/%5EIXIC?p=^IXIC
        self.Nasdaq = '^IXIC'

        print("BUILDING A LINEAR REGRESSION MODEL FOR THE FOLLOWING STOCKS:")
        print("Facebook, Amazon, Apple, Netflix, Google")
        print(tickers)

        # Defining other variables that will be used for calculations
        # The following variables will be used to pull the prices of stock decided

        self.tickers_frame = None
        self.closing_prices = None
        self.Nasdaq_frame = None
        self.closing_Nasdaq = None
        self.all_prices = None

        # Window is set
        # The following variables will be used in the function of setting the window(frame of n days)

        self.window = None
        self.all_prices_samples = None

        # The following variables will be used to calculate the variance
        self.window_means = None
        self.window_variance = None
        self.variance_func = None

        self.window_covariance = None

        self.window_slopes = None

        self.window_intercept = None

        self.Rsqrd = None
        self.Rsquared_series = None
        self.residual_square = None
        self.R_squared = None

        self.stderr_series = None

        self.t_scores = None

        self.p_val = None

    # pulling the prices of the stocks for the specified dates from Yahoo Finance API
    def PullPrices(self, startdate, enddate):

        print(
            "_____________________________________________________________________________")

        print("Pulling prices from Yahoo Finance API")
        print("Taking the time period as: " + str(startdate) +
              str(enddate))
        self.tickers_frame = pd.DataFrame(yhf.Tickers(
            self.tickers).history(start=startdate, end=enddate))
        self.closing_prices = self.tickers_frame['Close'].dropna(axis=1)
        self.Nasdaq_frame = pd.DataFrame(yhf.Ticker(
            self.Nasdaq).history(start=startdate, end=enddate))
        self.closing_Nasdaq = self.Nasdaq_frame['Close']
        self.all_prices = pd.concat(
            [self.closing_prices, self.closing_Nasdaq], axis=1)
        return self.all_prices

    # pull the prices of the stocks for the specified dates & window

    def DefineWindow(self, window):
        # Sample from the dataset using specified moving window
        # Create a list of subsets of the closing prices for the tickers and NASDAQ
        # The window of 30 days is set to get the data and perform the calculations

        print(
            "_____________________________________________________________________________")

        print("Let's set a window of: " + str(window) + " days. ")
        self.window = window
        self.all_prices_samples = []
        for i in range(window, len(self.closing_Nasdaq)):
            self.all_prices_samples.append(self.all_prices[(i - window):i])
        print("Dataset window size of: " + str(window))
        return self.all_prices_samples

    def CalculateVariance(self):
        # Calculate the mean values for each stock in each window
        # Store these means as pandas series objects

        print(
            "_____________________________________________________________________________")

        print("Let's calculate the Variance for the closing prices of Nasdaq.")
        # Calculate means first to calculate the variance
        self.window_means = []
        for i in range(0, len(self.all_prices_samples)):
            self.window_means.append(self.all_prices_samples[i].apply(np.mean))

        # Calculate the variance of NASDAQ closing prices for each window
        self.window_variance = []
        self.variance_func = lambda x: (x - self.window_means[i]['Close']) ** 2
        for i in range(0, len(self.all_prices_samples)):
            mean_x = self.window_means[i]['Close']
            # Formula to calculate variance
            var = sum(self.all_prices_samples[i]['Close'].map(self.variance_func)) / (
                len(self.all_prices_samples[i]['Close']) - 1)
            self.window_variance.append(var)
        return self.window_variance

    def CalculateCovariance(self):
        # Calculate the covariances of each stock and Nasdaq within each window

        print(
            "_____________________________________________________________________________")

        print("Let's calculate the Covariance for the closing prices of Nasdaq. ")
        def cov_x(x): return (x - self.window_means[i]['Close'])
        def cov_y(y): return (y - self.window_means[i][j])
        self.window_covariance = []
        for i in range(0, len(self.all_prices_samples)):
            cov_list = []
            index_list = []
            for j in self.all_prices_samples[i].drop(labels='Close', axis=1).columns:
                # Formula to calculate covariance
                covariance = sum(self.all_prices_samples[i][j].map(cov_y) *
                                 self.all_prices_samples[i]['Close'].map(cov_x)) / \
                    (len(self.all_prices_samples[i]['Close']) - 1)
                cov_list.append(covariance)
                index_list.append(j)
            self.window_covariance.append(
                pd.Series(cov_list, index=index_list))
        return self.window_covariance

    def CalculateSlopes(self):
        # Get the slope of the regression lines for each stock in each window
        # The slope is the same as the beta value

        print(
            "_____________________________________________________________________________")

        print("Let's Calculate the slopes")
        self.window_slopes = []
        for i in range(0, len(self.all_prices_samples)):
            self.window_slopes.append(
                self.window_covariance[i] / self.window_variance[i])
        # print(self.window_slopes)
        return self.window_slopes

    def CalculateIntercepts(self):
        # Get the intercept values for each stock in each window

        print(
            "_____________________________________________________________________________")

        print("Let's calculate the Intercepts")
        self.window_intercept = []
        for i in range(0, len(self.all_prices_samples)):
            intercept_List = []
            index_list = []
            w1 = self.window_slopes[i]
            mean_close = np.mean(self.all_prices_samples[i]['Close'])
            for j in self.all_prices_samples[i].drop(labels='Close', axis=1).columns:
                w0 = np.mean(
                    self.all_prices_samples[i][j]) - w1[j] * mean_close
                intercept_List.append(w0)
                index_list.append(j)
            self.window_intercept.append(
                pd.Series(intercept_List, index=index_list))
        # print(self.window_intercept[0])
        return self.window_intercept

    def PlottingWindow(self, windownum, tick):
        # Demonstrating that each set of points was divided into 30 day windows
        # and each window has a matching line function for each stock

        print(
            "_____________________________________________________________________________")
        print("Plotting window: " + str(windownum) + " of stock: " + str(tick))
        plt.scatter(self.all_prices_samples[windownum]['Close'],
                    self.all_prices_samples[windownum][tick], c='black')
        x = list(self.all_prices_samples[windownum]['Close'])
        y = []
        for i in x:
            y1 = i * (self.window_slopes[windownum][tick]) + \
                self.window_intercept[windownum][tick]
            y.append(y1)
        title_temp = [tick, 'vs Nasdaq for Window', str(windownum)]
        title = ' '.join(title_temp)
        # plotted a graph for the stock closing price vs NASDAQ closing price
        plt.title(title)
        plt.plot(x, y, c='blue')

    # END of liner regression package
    # Now Calculating R squared, Standard Error of Coefficient, T score and P value
    # To test model FIT

    def CalculateRsquared(self, windownum, list_tickers):
        # Calculate the R-squared value for a specific stock in a given window

        print(
            "_____________________________________________________________________________")
        print("The Calculated R-Squared value for a window of " +
              str(windownum) + str(list_tickers))
        self.Rsqrd = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            tot_squa = sum((self.all_prices_samples[windownum][list_tickers] - self.window_means[windownum][
                list_tickers]) ** 2)
            x = list(self.all_prices_samples[windownum]['Close'])
            y = []
            for i in x:
                y1 = i * (self.window_slopes[windownum][list_tickers]) + \
                    self.window_intercept[windownum][list_tickers]
                y.append(y1)
        # Formula to calculate the residual square
            residual_square = sum(
                (self.all_prices_samples[windownum][list_tickers] - y) ** 2)
            R_squared = 1 - (residual_square / tot_squa)
            self.Rsqrd.append(R_squared)
            indexes.append(list_tickers)
        self.Rsquared_series = pd.Series(self.Rsqrd, index=indexes)
        return self.Rsquared_series

    def CalculateStdcoeff(self, windownum, list_tickers):
        # Get the standard error coefficients

        print(
            "_____________________________________________________________________________")
        print("the calculated Standard Error Coefficients for a window of " +
              str(windownum) + str(list_tickers))
        serror = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            n = len(self.all_prices_samples[windownum])
            # Formula to calculate Standard coefficient
            se_coef = (math.sqrt(sum((self.all_prices_samples[windownum][list_tickers] -
                                      self.all_prices_samples[windownum]['Close']) ** 2) / (n - 2)) / math.sqrt(
                sum((self.all_prices_samples[windownum]['Close'] - self.window_means[windownum]['Close']) ** 2)))
            serror.append(se_coef)
            indexes.append(list_tickers)
        self.stderr_series = pd.Series(serror, index=indexes)
        return self.stderr_series

    def CalculateTscores(self, windownum, list_tickers):
        # Get the t-scores

        print(
            "_____________________________________________________________________________")
        print("The calculated T score for a window of " +
              str(windownum) + str(list_tickers))
        tsc = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            t = self.window_slopes[windownum][list_tickers] / \
                self.stderr_series[list_tickers]
            tsc.append(t)
            indexes.append(list_tickers)
            # Tscore value is taken from Numpy Lib
        self.t_scores = pd.Series(tsc, index=indexes)
        return self.t_scores

    def CalculatePvalues(self, list_tickers):

        # scipy stats package is used to calculate the p values

        print(
            "_____________________________________________________________________________")
        print("The calculated P values for tickers: " + str(list_tickers))
        df = len(self.all_prices_samples) - 2
        self.p_val = 1 - stat.t.cdf(self.t_scores[list_tickers], df=df)
        temp_df = pd.Series(self.p_val, index=list_tickers)
        return temp_df


# In[7]:


#

# In[ ]:
