"""
Test Project
RUN EVERYTHING FOR THE REPORT

"""

import datetime as dt
import random
import numpy as np
import pandas as pd
import util as ut
import indicators as indi
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt
import StrategyLearner as sl
import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2


def testcode():

    #set seed
    np.random.seed(903748900)

    #MANUAL STRATEGY
    #inputs
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sd_test = dt.datetime(2010, 1, 1)
    ed_test = dt.datetime(2011, 12, 31)
    symbol = "JPM"

    #Create auto-trade order based on manual set of rules. See Manual Strategy.py
    #create benchmark order, buy and hold until the end of period
    manual = ms.ManualStrategy()
    trades_df = manual.testPolicy(symbol,sd,ed,sv)
    benchmark_df = manual.test_benchmark(symbol,sd,ed,sv)

    #Calculate portfolio value
    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)
    trades_df['position'] = trades_df[symbol].cumsum()

    ms.plot_graph(benchmark_portvals,trades_portvals,trades_df)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("FOR IN-SAMPLE MANUAL STRATEGY")
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

    #for out sample test, repeat step above
    trades_df = manual.testPolicy(symbol,sd_test,ed_test,sv)
    benchmark_df = manual.test_benchmark(symbol,sd_test,ed_test,sv)
    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)
    trades_df['position'] = trades_df[symbol].cumsum()
    ms.plot_graph_out_sample(benchmark_portvals,trades_portvals,trades_df)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("FOR OUT-SAMPLE MANUAL STRATEGY")
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


    ### END OF MANUAL STRATEGY
    ### STRATEGY LEARNER

    #See StrategyLearner.py for details:
    #Strategy learner is bagging of random forest learner
    # with leaves size =5 and bags = 10
    # 3 feature indicators in consideration is:
    # EMA9 CROSS EMA21 or EMA9-EMA21 difference
    # BOLLINGER BAND PERCENTAGE
    # MACD HISTOGRAM
    strategy = sl.StrategyLearner()
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase

    # calculate portfolio value
    trades_portvals_SL = compute_portvals(trades_df_SL, sv, commission = 9.95, impact = 0.005)
    trades_df_SL['position'] = trades_df_SL[symbol].cumsum()

    sl.plot_graph_SL(trades_portvals_SL,trades_df_SL) #plot graph

    #create table result
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals_SL)

    print("STRATEGY LEARNER - IN SAMPLE TEST")
    print(f"Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Strategy: ", cum_ret)
    print("Avg daily Return: ")
    print("Strategy: ", avg_daily_ret)
    print("Std daily Return: ")
    print("Strategy: ", std_daily_ret)
    print("Sharpe Ratio: ")
    print("Strategy: ", sharpe_ratio)

    ### OF STRATEGY LEARNER
    ### EXPERIMENT 1: Compare Manual vs Strategy Learner

    # # in sample comparison
    # trades_df = manual.testPolicy(symbol,sd,ed,sv)
    # benchmark_df = manual.test_benchmark(symbol,sd,ed,sv)
    exp1.testcode()
    exp2.testcode()

def author():
    return ''

if __name__ == "__main__":
    print("EVERYTHING IN ONE")
    testcode()