"""

MANUAL STRATEGY

testpolicy should return trades data frame
trades_df should have index as 'Date','Symbol','Order','Shares'

current position limit 1000 shares long or short
Commission 9.95, impact 0.005

Manual strategy, use 9x21 for entry, use MACD histogram or BBP for confirmation
If 9x21 > 0 and either macd hist >=0 or BBP >=0.5, go long
exit when both MACD hist <=0 and BBP <=0.5
reentry position of MACD or BBP condition get better.
For short, vice versa

ema9x21, macd_hist, bbp
---------------------------------------
Student Name: Trung Pham (replace with your name)
GT User ID: tpham328 (replace with your User ID)
GT ID: 903748900 (replace with your GT ID)
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import indicators as indi
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt

class ManualStrategy:

    def testPolicy(self, symbol ="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   sv=100000):

        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna(how='any', subset=[symbol])
        prices = prices.ffill().bfill()
        prices.to_csv('prices.txt', sep='\t', index=True)

        trades_df = pd.DataFrame(columns = ['Date',symbol])
        trades_df = trades_df.set_index('Date')

        #Calculate indicators
        ema9x21 = indi.ema9xema21(symbol, sd, ed)[2]
        bbp = indi.bbp(symbol, sd, ed)[3]
        macd_hist = indi.MACD(symbol,sd,ed)[2]
        ema9x21.to_csv('ema9x21.txt', sep ='\t', index=True)
        bbp.to_csv('bbp.txt', sep='\t', index=True)
        macd_hist.to_csv('macd_hist.txt', sep='\t', index=True)

        # Initialize current position
        current_position = 0

        #loop through each date in the date range and apply stratagy
        for i in range(len(prices)-19):
            date = prices.index[i+19]
            # price = prices.at[prices.index]
            vote = 0 #reset vote
            if ema9x21.at[prices.index[i+19],symbol] >= 0 and \
                    bbp.at[prices.index[i+19],symbol] < 70 and \
                    macd_hist.at[prices.index[i+19],symbol] >= 0:
                vote = 1
            elif ema9x21.at[prices.index[i+19],symbol] < 0 and \
                    bbp.at[prices.index[i+19],symbol] > 30 and \
                    macd_hist.at[prices.index[i+19],symbol] < 0:
                vote = -1
            else:
                vote = 0

            #active trade
            if current_position == 0:
                if vote == 1:
                    trades_df.loc[date] = [1000]
                    current_position = 1000
                    is_long = True
                if vote == -1:
                    trades_df.loc[date] = [-1000]
                    current_position = -1000
                    is_long = False
            if current_position == 1000:
                if vote == -1:
                    trades_df.loc[date] = [-2000]
                    current_position = current_position - 2000
                if vote == 0:
                    trades_df.loc[date] = [-1000]
                    current_position = current_position - 1000
            if current_position == -1000:
                if vote == 1:
                    trades_df.loc[date] = [2000]
                    current_position = current_position + 2000
                if vote == 0:
                    trades_df.loc[date] = [1000]
                    current_position = current_position + 1000

        trades_df.to_csv('trades_df.txt', sep='\t', index=True)

        return trades_df


    #From project 6, benchmark buy and hold position during test period
    def test_benchmark(self,symbol, sd, ed, sv=100000):
        benchmark_df = pd.DataFrame(columns=['Date', symbol])
        benchmark_df = benchmark_df.set_index('Date')
        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna(how='any', subset=[symbol])
        benchmark_df.loc[prices.index[0]] = [1000]
        benchmark_df.loc[prices.index[-1]] = [-1000]
        benchmark_df.to_csv('benchmark_df.txt', sep='\t', index=True)
        return benchmark_df

def plot_graph(benchmark_portvals, trades_portvals,trades_df):

    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]
    trades_portvals = trades_portvals/trades_portvals[0]

    plt.plot(trades_portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    for date, value in trades_df.loc[trades_df['position'] > 0].iterrows():
        plt.axvline(x=date, color = "blue", alpha = 0.3, linestyle ='dashed')
    for date, value in trades_df.loc[trades_df['position'] < 0].iterrows():
        plt.axvline(x=date, color = "black", alpha = 0.3, linestyle ='dashed')
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Manual Strategy Vs Benchmark (in sample)")
    plt.savefig("ManualStr_in_sample.png")
    plt.close()

def plot_graph_out_sample(benchmark_portvals, trades_portvals,trades_df):

    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]
    trades_portvals = trades_portvals/trades_portvals[0]

    plt.plot(trades_portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    for date, value in trades_df.loc[trades_df['position'] > 0].iterrows():
        plt.axvline(x=date, color = "blue", alpha = 0.3, linestyle ='dashed')
    for date, value in trades_df.loc[trades_df['position'] < 0].iterrows():
        plt.axvline(x=date, color = "black", alpha = 0.3, linestyle ='dashed')
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Manual Strategy Vs Benchmark (out sample)")
    plt.savefig("ManualStr_out_sample.png")
    plt.close()

def test_code():
    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"

    ms = ManualStrategy()
    trades_df = ms.testPolicy(symbol,sd,ed,sv)
    benchmark_df = ms.test_benchmark(symbol,sd,ed,sv)

    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)
    trades_df['position'] = trades_df[symbol].cumsum()

    # export for debug
    # trades_df.to_csv('trades_df.test.txt', sep='\t', index=True)
    # trades_portvals.to_csv('trades_portvals.txt', sep='\t', index=True)
    # benchmark_portvals.to_csv('benchmark_portvals.txt', sep='\t', index=True)

    plot_graph(benchmark_portvals,trades_portvals,trades_df)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("IN SAMPLE")
    print(f"Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Manual: ", cum_ret)
    print("Benchmark: ", ben_cum_ret)
    print("Avg daily Return: ")
    print("Manual: ", avg_daily_ret)
    print("Benchmark: ", ben_avg_daily_ret)
    print("Std daily Return: ")
    print("Manual: ", std_daily_ret)
    print("Benchmark: ", ben_std_daily_ret)
    print("Sharpe Ratio: ")
    print("Manual: ", sharpe_ratio)
    print("Benchmark: ", ben_sharpe_ratio)

    #TEST OUT-SAMPLE. repeat above steps
    sd = dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)

    ms = ManualStrategy()
    trades_df = ms.testPolicy(symbol,sd,ed,sv)
    benchmark_df = ms.test_benchmark(symbol,sd,ed,sv)

    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)
    trades_df['position'] = trades_df[symbol].cumsum()

    plot_graph_out_sample(benchmark_portvals,trades_portvals,trades_df)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("OUT SAMPLE")
    print(f"Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Manual: ", cum_ret)
    print("Benchmark: ", ben_cum_ret)
    print("Avg daily Return: ")
    print("Manual: ", avg_daily_ret)
    print("Benchmark: ", ben_avg_daily_ret)
    print("Std daily Return: ")
    print("Manual: ", std_daily_ret)
    print("Benchmark: ", ben_std_daily_ret)
    print("Sharpe Ratio: ")
    print("Manual: ", sharpe_ratio)
    print("Benchmark: ", ben_sharpe_ratio)


def author():
    return 'tpham328'

if __name__ == "__main__":
    test_code()




