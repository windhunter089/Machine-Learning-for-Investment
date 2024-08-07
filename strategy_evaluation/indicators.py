"""
Student Name: Trung Pham (replace with your name)
GT User ID: tpham328 (replace with your User ID)
GT ID: 903748900 (replace with your GT ID)
"""

"""
5 indicator:
EMA
Bollinger Bands %
Stochastic Indicator
MACD
Rate of chanage (ROC)
"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt

def author():
    return 'tpham328'

def ema9xema21(symbol,sd,ed):
    ema9 = ema(symbol,sd,ed, lookback=9)
    ema21 = ema(symbol,sd,ed, lookback=21)

    ema9x21 = ema9 - ema21

    return ema9,ema21,ema9x21

def MACD(symbol,sd,ed):
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])

    ema12 = prices.ewm(span=12,adjust=False).mean()
    ema26 = prices.ewm(span=26,adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9,adjust=False).mean()

    macd_hist = macd_line - signal_line

    return macd_line,signal_line, macd_hist

def bbp(symbol,sd,ed,lookback = 20):


    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])

    rolling_mean = prices.rolling(window=lookback,min_periods=lookback).mean()
    rolling_std = prices.rolling(window=lookback,min_periods=lookback).std()
    top_band = rolling_mean + 2 * rolling_std
    bottom_band = rolling_mean - 2 * rolling_std

    bbq = (prices - bottom_band) / (top_band - bottom_band) * 100

    return rolling_mean,top_band,bottom_band, bbq

"""
Use the following formula from stockcharts.com
%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
%D = 3-day SMA of %K

Lowest Low = lowest low for the look-back period
Highest High = highest high for the look-back period
%K is multiplied by 100 to move the decimal point two places
"""
def stochastic(symbol,sd,ed, k_period = 14, d_period = 3):

    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
    prices = prices.dropna(how='any', subset=[symbol])
    prices_high = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="High")
    prices_high = prices_high.dropna(how='any', subset=[symbol])
    prices_low = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="Low")
    prices_low = prices_low.dropna(how='any', subset=[symbol])

    k = pd.DataFrame(columns=['Date', symbol])
    k = k.set_index('Date')

    for i in range(len(prices) - k_period + 1): #20-14 +1 = 7, range (7) 0-6
        date = prices.index[i+k_period-1]
        price_high = prices_high.iloc[i:i+k_period]
        price_low = prices_low.iloc[i:i+k_period]

        highest_high = price_high[symbol].max()
        # print(highest_high)
        lowest_low = price_low[symbol].min()
        # print(lowest_low)
        k.loc[date] = (prices.iloc[i+k_period-1][symbol] - lowest_low) / (highest_high - lowest_low) * 100

    d = k.rolling(window=d_period, min_periods=d_period).mean()
    # print(k.head(10))
    # print(d)
    return k,d

def rate_of_change(symbol,sd,ed,lookback=5):
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])

    roc = (prices / prices.shift(lookback)) - 1
    roc = roc * 100

    # print(roc.head(10))
    return roc

"""
Helper
"""

def ema(symbol,sd=dt.datetime(2008,1,1),ed=dt.date(2009,12,31),lookback = 9):

    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])
    ema = prices.ewm(span = lookback, adjust=False).mean()

    return ema

"""
End Helper
"""


def test_code():

    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"

    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])
    # print(prices.head(10))


    ema9x21 = ema9xema21(symbol,sd,ed)
    # print(ema9x21.head(10))

    rolling_mean, top_band, bottom_band, bbq = bbp(symbol,sd,ed)

    k,d = stochastic(symbol,sd,ed, k_period = 14, d_period = 3)

    roc = rate_of_change(symbol,sd,ed)

if __name__ == "__main__":
    test_code()