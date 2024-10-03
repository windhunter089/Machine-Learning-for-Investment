"""
Experiment 1: Compare result of manual strategy with strategy learner

Compare in-sample and out of sample JPM


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


def plot_in_sample(benchmark_portvals, trades_portvals,trades_portvals_SL):

    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]
    trades_portvals = trades_portvals/trades_portvals[0]
    trades_portvals_SL = trades_portvals_SL/trades_portvals_SL[0]
    plt.plot(trades_portvals_SL, label="Strategy Learner", color="green")
    plt.plot(trades_portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Manual Strategy Vs Strategy Learner (in sample)")
    plt.savefig("exp1_in_sample.png")
    plt.close()

def plot_out_sample(benchmark_portvals, trades_portvals,trades_portvals_SL):

    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]
    trades_portvals = trades_portvals/trades_portvals[0]
    trades_portvals_SL = trades_portvals_SL/trades_portvals_SL[0]
    plt.plot(trades_portvals_SL, label="Strategy Learner", color="green")
    plt.plot(trades_portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Manual Strategy Vs Strategy Learner (out sample)")
    plt.savefig("exp1_out_sample.png")
    plt.close()

def testcode():
    np.random.seed(903748900)

    #in-sample data inputs
    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"

    manual = ms.ManualStrategy()
    trades_df = manual.testPolicy(symbol,sd,ed,sv)
    benchmark_df = manual.test_benchmark(symbol,sd,ed,sv)


    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)

    strategy = sl.StrategyLearner()
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase

    trades_portvals_SL = compute_portvals(trades_df_SL, sv, commission = 9.95, impact = 0.005)

    plot_in_sample(benchmark_portvals,trades_portvals,trades_portvals_SL)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL = port_stats(trades_portvals_SL)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("EXPERIMENT 1- IN SAMPLE TEST")
    print(f"IN SAMPLE Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Manual: ", cum_ret)
    print("Strategy: ", cum_ret_SL)
    print("Benchmark: ", ben_cum_ret)
    print("Avg daily Return: ")
    print("Manual: ", avg_daily_ret)
    print("Strategy: ", avg_daily_ret_SL)
    print("Benchmark: ", ben_avg_daily_ret)
    print("Std daily Return: ")
    print("Manual: ", std_daily_ret)
    print("Strategy: ", std_daily_ret_SL)
    print("Benchmark: ", ben_std_daily_ret)
    print("Sharpe Ratio: ")
    print("Manual: ", sharpe_ratio)
    print("Strategy: ", sharpe_ratio_SL)
    print("Benchmark: ", ben_sharpe_ratio)
    """
    Create out of sample test
    """
    sd = dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)

    manual = ms.ManualStrategy()
    trades_df = manual.testPolicy(symbol,sd,ed,sv)
    benchmark_df = manual.test_benchmark(symbol,sd,ed,sv)


    trades_portvals = compute_portvals(trades_df, sv, commission = 9.95, impact = 0.005)
    benchmark_portvals = compute_portvals(benchmark_df, sv, commission=9.95, impact = 0.005)

    strategy = sl.StrategyLearner()
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase

    trades_portvals_SL = compute_portvals(trades_df_SL, sv, commission = 9.95, impact = 0.005)

    plot_out_sample(benchmark_portvals,trades_portvals,trades_portvals_SL)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals)
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL = port_stats(trades_portvals_SL)
    ben_cum_ret, ben_avg_daily_ret, ben_std_daily_ret,ben_sharpe_ratio = port_stats(benchmark_portvals)

    print("EXPERIMENT 1: OUT-SAMPLE TEST")
    print(f"OUT SAMPLE Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Manual: ", cum_ret)
    print("Strategy: ", cum_ret_SL)
    print("Benchmark: ", ben_cum_ret)
    print("Avg daily Return: ")
    print("Manual: ", avg_daily_ret)
    print("Strategy: ", avg_daily_ret_SL)
    print("Benchmark: ", ben_avg_daily_ret)
    print("Std daily Return: ")
    print("Manual: ", std_daily_ret)
    print("Strategy: ", std_daily_ret_SL)
    print("Benchmark: ", ben_std_daily_ret)
    print("Sharpe Ratio: ")
    print("Manual: ", sharpe_ratio)
    print("Strategy: ", sharpe_ratio_SL)
    print("Benchmark: ", ben_sharpe_ratio)




def author():
    return ''

if __name__ == "__main__":
    print("Exp1")
    testcode()
