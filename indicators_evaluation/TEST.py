"""

"""

"""
Implementing best trades strategy
testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
return a trades data frame (best trades order to get optimal return
trades df: 'Date','Symbol','Order','Shares'
"""

"""
Can only long or short 1000 shares, starting value 100,000
Rules: if price next day go up, go long today.
If price next day go down, go short today.
Long short price is adj closing price today
"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt

def author():
    return ''


def testPolicy(symbol ="JPM",
                sd=dt.datetime(2008,1,1),
                ed=dt.datetime(2009,12,31),
                sv = 100000):

    prices = get_data([symbol], pd.date_range(sd,ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])
    prices = prices.ffill().bfill()
    # prices.to_csv('prices.txt', sep='\t', index=True)

    trades_df = pd.DataFrame(columns = ['Date',symbol])
    trades_df = trades_df.set_index('Date')
    # Initialize current position
    current_position = 0
    is_long = False

    # loop through each date in price and apply the rule/strategy
    for i in range(len(prices)-1):
        date = prices.index[i]
        price = prices.at[prices.index[i],symbol]
        # print(date)
        # print(price)
        if current_position == 0:
            if prices.at[prices.index[i+1],symbol] > price:
                #tomorrow price is higher, buy
                trades_df.loc[date] = [1000]
                current_position = 1000
                is_long = True
            elif prices.at[prices.index[i+1],symbol] < price:
                #tomorrow price is lower, sell
                trades_df.loc[date] = [-1000]
                current_position = -1000
                is_long = False
        elif is_long:
            if prices.at[prices.index[i+1],symbol] < price:
                #tomorrow price is lower, sell current position and short another 1000
                trades_df.loc[date] = [-2000]
                current_position = current_position - 2000
                is_long = False
        else:
            if prices.at[prices.index[i+1],symbol] > price:
                #tomorrow price is higher, buy back short and long another 1000 shares
                trades_df.loc[date] = [2000]
                current_position = current_position + 2000
                is_long = True

    # trades_df.to_csv('trades_df.txt', sep='\t', index=True)
    return trades_df

def test_benchmark(symbol, sd, ed, sv):
    benchmark_df = pd.DataFrame(columns=['Date', symbol])
    benchmark_df = benchmark_df.set_index('Date')
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbol])
    benchmark_df.loc[prices.index[0]] = [1000]
    benchmark_df.loc[prices.index[-1]] = [-1000]
    # benchmark_df.to_csv('benchmark_df.txt', sep='\t', index=True)
    return benchmark_df

def plot_Part1(benchmark_portvals, test_portvals):

    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]
    test_portvals = test_portvals/test_portvals[0]

    plt.plot(test_portvals, label="TOS", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("TOS Vs Benchmark")
    plt.savefig("TOS.png")
    plt.close()

def test_code():

    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"

    trades_df = testPolicy(symbol,sd,ed,sv)
    benchmark_df = test_benchmark(symbol,sd,ed,sv)

    trades_portvals = compute_portvals(trades_df, sv, commission = 0.0, impact = 0.0)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=0.0, impact=0.0)

    # trades_portvals.to_csv('trades_portvals.txt', sep='\t', index=True)
    # benchmark_portvals.to_csv('benchmark_portvals.txt', sep='\t', index=True)

    plot_Part1(benchmark_portvals,trades_portvals)

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

if __name__ == "__main__":
        test_code()