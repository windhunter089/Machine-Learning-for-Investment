"""
Student Name: Trung Pham (replace with your name)
GT User ID: tpham328 (replace with your User ID)
GT ID: 903748900 (replace with your GT ID)
"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt
import indicators as indi
import TheoreticallyOptimalStrategy as tos


def test_code():

    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"

    """
    Part 1- Theoretically Optimal Strategy
    """
    trades_df = tos.testPolicy(symbol,sd,ed,sv)
    benchmark_df = tos.test_benchmark(symbol,sd,ed,sv)

    trades_portvals = compute_portvals(trades_df, sv, commission = 0.0, impact = 0.0)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=0.0, impact=0.0)
    # trades_portvals.to_csv('trades_portvals.txt', sep='\t', index=True)
    # benchmark_portvals.to_csv('benchmark_portvals.txt', sep='\t', index=True)

    tos.plot_Part1(benchmark_portvals,trades_portvals)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret, ben_sharpe_ratio = port_stats(benchmark_portvals)

    print(f"Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("TOS: ", cum_ret)
    print("Benchmark: ", ben_cum_ret)
    print("Avg daily Return: ")
    print("TOS: ", avg_daily_ret)
    print("Benchmark: ", ben_avg_daily_ret)
    print("Std daily Return: ")
    print("TOS: ", std_daily_ret)
    print("Benchmark: ", ben_std_daily_ret)
    print("Sharpe Ratio: ")
    print("TOS: ", sharpe_ratio)
    print("Benchmark: ", ben_sharpe_ratio)

    """
    Part 2 - Indicators:
    EMA crossover
    Bollinger Band Percentage
    MACD
    Stochastic Oscillator
    Rate of Change
    """
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])
    # print(prices.head(10))

    #EMA9 Cross EMA21 signal, >0 is BUY, <0 is SELL
    ema9,ema21,ema9x21 = indi.ema9xema21(symbol,sd,ed)
    # print(ema9x21.head(10))
    #PLOT EMA9 X EMA21
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(prices, label="price", color="darkred")
    ax1.plot(ema9, label="EMA9", color="gold")
    ax1.plot(ema21, label="EMA21", color="deepskyblue")
    ax1.grid(True,axis="both")
    ax2.plot(ema9x21, label="ema9X21", color="black")
    ax2.grid(True,axis="both")
    ax2.axhline(0, color = "skyblue")
    ax1.set_ylabel("Price of {}".format(symbol))
    ax2.set_ylabel("EMA9 x EMA21 difference")
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.subplots_adjust(hspace=0.2)
    plt.xticks(rotation=30)
    plt.suptitle("EMA9 cross EMA21 strategy")
    plt.xlabel("Date")
    plt.savefig("ema9x21.png")
    plt.close()

    #PLOT BOLLINGER BAND
    #price below bb band is buy signal, above bb band is sell signal
    #Another way to get signal is use bb percentage, bb percentage below 20 is buy, above 80 is sell
    rolling_mean, top_band, bottom_band, bbq = indi.bbp(symbol,sd,ed)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(prices, label="price", color="darkred")
    ax1.plot(top_band, label="top BB", color="deepskyblue")
    ax1.plot(bottom_band, label="bottom BB", color="deepskyblue")
    ax1.grid(True,axis="both")
    ax2.plot(bbq, label="BollingerBandPercent", color="black")
    ax2.grid(True,axis="both")
    ax2.axhline(20, color = "lime")
    ax2.axhline(80, color = "red")
    ax1.set_ylabel("Price of {}".format(symbol))
    ax2.set_ylabel("Bollinger Band Percentage")
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.subplots_adjust(hspace=0.2)
    plt.xticks(rotation=30)
    plt.suptitle("Bollinger Band Percentage Strategy")
    plt.xlabel("Date")
    plt.savefig("bbp.png")
    plt.close()

    # PLOT MACD
    # signal to buy and sell is when MACD line cut signal line
    # Cross over is buy, cross under is sell, combine into 1 vector histogram and use cross 0 to signal
    macd_line,signal_line,macd_hist = indi.MACD(symbol,sd,ed)
    # print(macd_hist.head(10))
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(prices, label="price", color="darkred")
    ax2.plot(macd_line, label="macd_line", color="orange")
    ax2.plot(signal_line, label="signal_line", color="royalblue")
    ax2.bar(macd_hist.index,macd_hist[symbol], width=1,color=['g' if h > 0 else 'r' for h in macd_hist[symbol]],alpha = 0.5)
    ax1.grid(True,axis="both")
    ax2.grid(True,axis="both")
    ax1.set_ylabel("Price of {}".format(symbol))
    ax2.set_ylabel("MACD")
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.subplots_adjust(hspace=0.2)
    plt.xticks(rotation=30)
    plt.suptitle("MACD Strategy")
    plt.xlabel("Date")
    plt.savefig("MACD.png")
    plt.close()

    # PLOT Stochastic Oscillator
    # when k line cross d line, use it as signal
    # Cross over is buy, cross under is sell, prefer to use when cross a threshold (30 & 70th percentile)
    k,d = indi.stochastic(symbol,sd,ed, k_period = 14, d_period = 3)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(prices, label="price", color="darkred")
    ax2.plot(k, label="k_line", color="orange")
    ax2.plot(d, label="d_line", color="royalblue")
    ax1.grid(True,axis="both")
    ax2.grid(True,axis="both")
    ax1.set_ylabel("Price of {}".format(symbol))
    ax2.set_ylabel("Stochastic Oscillator")
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.subplots_adjust(hspace=0.2)
    plt.xticks(rotation=30)
    plt.suptitle("Stochastic Strategy")
    plt.xlabel("Date")
    plt.savefig("Stochastic.png")
    plt.close()

    # PLOT Rate of Change Indicator
    # signal to buy and sell is when ROC is extreme value
    # High positive is buy, and low negative is sell
    roc = indi.rate_of_change(symbol,sd,ed)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(prices, label="price", color="darkred")
    ax2.plot(roc, label="ROC", color="orange")
    ax1.grid(True,axis="both")
    ax2.grid(True,axis="both")
    ax1.set_ylabel("Price of {}".format(symbol))
    ax2.set_ylabel("Rate of Change in %")
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.subplots_adjust(hspace=0.2)
    plt.xticks(rotation=30)
    plt.suptitle("Rate of Change Indicators")
    plt.xlabel("Date")
    plt.savefig("roc.png")
    plt.close()

if __name__ == "__main__":
    test_code()