"""
Experiment 2: Compare result of manual strategy with strategy learner

Measure how impact affect in-sample trading behavior


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


def plot_graph(trades_portvals1, trades_portvals2, trades_portvals3):

    trades_portvals1 = trades_portvals1/trades_portvals1.iloc[0]
    trades_portvals2 = trades_portvals2/trades_portvals2[0]
    trades_portvals3 = trades_portvals3/trades_portvals3[0]
    plt.plot(trades_portvals1, label="impact = 0", color="green")
    plt.plot(trades_portvals2, label="impact = 0.005", color="red")
    plt.plot(trades_portvals3, label="impact = 0.05", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Strategy learner performance under different impact")
    plt.savefig("exp2.png")
    plt.close()
def testcode():
    np.random.seed(903748900)

    #in-sample data inputs
    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "JPM"


    #Create learner, train the model
    #Calculate portfolio value under different impact value 0.005
    strategy = sl.StrategyLearner(verbose=False,impact=0.000)
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase
    trades_portvals1 = compute_portvals(trades_df_SL, sv, commission = 0, impact = 0.000)

    strategy = sl.StrategyLearner(verbose=False,impact=0.005)
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase
    trades_portvals2 = compute_portvals(trades_df_SL, sv, commission = 0, impact = 0.005)

    strategy = sl.StrategyLearner(verbose=False,impact=0.05)
    strategy.add_evidence(symbol,sd,ed,sv)  # training phase
    trades_df_SL = strategy.testPolicy(symbol,sd,ed,sv)  # testing phase
    trades_portvals3 = compute_portvals(trades_df_SL, sv, commission = 0, impact = 0.05)

    plot_graph(trades_portvals1, trades_portvals2, trades_portvals3)

    cum_ret1, avg_daily_ret1, std_daily_ret1, sharpe_ratio1 = port_stats(trades_portvals1)
    cum_ret2, avg_daily_ret2, std_daily_ret2, sharpe_ratio2 = port_stats(trades_portvals2)
    cum_ret3, avg_daily_ret3, std_daily_ret3, sharpe_ratio3 = port_stats(trades_portvals3)

    print("EXPERIMENT 2: impact affects performance")
    print(f"IN SAMPLE Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("impact = 0: ", cum_ret1)
    print("impact = 0.005: ", cum_ret2)
    print("impact = 0.05: ", cum_ret3)
    print("Avg daily Return: ")
    print("impact = 0: ", avg_daily_ret1)
    print("impact = 0.005: ", avg_daily_ret2)
    print("impact = 0.05: ", avg_daily_ret3)
    print("Std daily Return: ")
    print("impact = 0: ", std_daily_ret1)
    print("impact = 0.005: ", std_daily_ret2)
    print("impact = 0.05: ", std_daily_ret3)
    print("Sharpe Ratio: ")
    print("impact = 0: ", sharpe_ratio1)
    print("impact = 0.005: ", sharpe_ratio2)
    print("impact = 0.05: ", sharpe_ratio3)

def author():
    return ''

if __name__ == "__main__":
    print("Exp2")
    testcode()
