""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Trung Pham (replace with your name)
GT User ID: tpham328 (replace with your User ID)
GT ID: 903748900 (replace with your GT ID)		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import datetime as dt  		  	   		  		 			  		 			     			  	 
import random  		  	   		  		 			  		 			     			  	 
import numpy as np
import pandas as pd  		  	   		  		 			  		 			     			  	 
import util as ut
import indicators as indi
from marketsimcode import compute_portvals, port_stats
import matplotlib.pyplot as plt
import RTLearner as rt
import BagLearner as bl
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class StrategyLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    # constructor  		  	   		  		 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
        self.impact = impact  		  	   		  		 			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(learner = rt.RTLearner,
                                kwargs = {"leaf_size": 5},
                                bags = 10,
                                boost = False,
                                verbose=False
                                )  # create bag learner RT

    def add_evidence(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        # add your code to do learning here
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna(how='any', subset=[symbol])
        prices = prices.ffill().bfill()
        prices.to_csv('prices.txt', sep='\t', index=True)

        # trades_df = pd.DataFrame(columns=['Date', symbol])
        # trades_df = trades_df.set_index('Date')

        # Calculate indicators
        ema9x21 = indi.ema9xema21(symbol, sd, ed)[2]
        bbp = indi.bbp(symbol, sd, ed)[3]
        macd_hist = indi.MACD(symbol, sd, ed)[2]
        ema9x21 = ema9x21.rename(columns={symbol: 'EMA9X21'})
        bbp = bbp.rename(columns={symbol:'BBP'})
        macd_hist = macd_hist.rename(columns={symbol: 'MACD_Hist'})
        ema9x21.to_csv('ema9x21.txt', sep='\t', index=True)
        bbp.to_csv('bbp.txt', sep='\t', index=True)
        macd_hist.to_csv('macd_hist.txt', sep='\t', index=True)

        #set N days to peek in future
        n = 10
        trainx = pd.concat([ema9x21, bbp, macd_hist],axis = 1)
        trainx = trainx[19:-n]
        trainx = trainx.values
        # np.savetxt('trainx.txt', trainx)
        trainy=np.zeros(trainx.shape[0])

        for i in range(prices.shape[0]-n-20):
            ret = (prices.iloc[i+n]/prices.iloc[i]) - 1.0
            if ret.values >= (0.02 + self.impact):
                trainy[i] = +1
            elif ret.values < (-0.02 - self.impact):
                trainy[i] = -1
            else:
                trainy[i] = 0
        trainy = np.array(trainy)
        # np.savetxt('trainy.txt', trainy)

        self.learner.add_evidence(trainx,trainy)


  		  	   		  		 			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  		 			  		 			     			  	 
    def testPolicy(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			     			  	 
        """
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna(how='any', subset=[symbol])
        prices = prices.ffill().bfill()
        prices.to_csv('prices.txt', sep='\t', index=True)

        # Calculate indicators
        ema9x21 = indi.ema9xema21(symbol, sd, ed)[2]
        bbp = indi.bbp(symbol, sd, ed)[3]
        macd_hist = indi.MACD(symbol, sd, ed)[2]
        ema9x21 = ema9x21.rename(columns={symbol: 'EMA9X21'})
        bbp = bbp.rename(columns={symbol: 'BBP'})
        macd_hist = macd_hist.rename(columns={symbol: 'MACD_Hist'})
        ema9x21.to_csv('ema9x21.txt', sep='\t', index=True)
        bbp.to_csv('bbp.txt', sep='\t', index=True)
        macd_hist.to_csv('macd_hist.txt', sep='\t', index=True)

        # set N days to peek in future
        N = 10
        testx = pd.concat([ema9x21, bbp, macd_hist], axis=1)
        testx = testx[19:-N]
        testx = testx.values
        # np.savetxt('testx.txt', testx)

        predy = []
        predy = self.learner.query(testx)
        trades_df = pd.DataFrame(columns = ['Date',symbol])
        trades_df = trades_df.set_index('Date')
        current_position = 0

        for i in range(len(prices)-N-20):
            date = prices.index[i]
            if current_position == 0:
                if predy[i] == 1:
                    trades_df.loc[date] = 1000
                    current_position = 1000
                elif predy[i] == -1:
                    trades_df.loc[date] = -1000
                    current_position = -1000
            elif current_position == 1000:
                if predy[i] == 0:
                    trades_df.loc[date] = -1000
                    current_position = current_position - 1000
                elif predy[i] == -1:
                    trades_df.loc[date] = -2000
                    current_position = current_position - 2000
            elif current_position == -1000:
                if predy[i] == 0:
                    trades_df.loc[date] = +1000
                    current_position = current_position + 1000
                elif predy[i] == 1:
                    trades_df.loc[date] = +2000
                    current_position = current_position + 2000

        return trades_df


def testcode():
    sl = StrategyLearner()
    sv = 100000
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = "AAPL"

    sl.add_evidence(symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                         sv=100000)  # training phase
    trades_df_SL = sl.testPolicy(symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                   sv=100000)  # testing phase
    # trades_df_SL.to_csv('trades_df_SL.txt', sep='\t', index=True)
    trades_df_SL['position'] = trades_df_SL[symbol].cumsum()

    trades_portvals_SL = compute_portvals(trades_df_SL, sv, commission = 9.95, impact = 0.005)
    # trades_portvals_SL.to_csv('trades_portvals_SL.txt', sep='\t', index=True)

    plot_graph_SL(trades_portvals_SL,trades_df_SL)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(trades_portvals_SL)

    print("STRATEGY LEARNER")
    print(f"Date Range: {sd} to {ed}")
    print("Cumulative Return: ")
    print("Strategy: ", cum_ret)
    print("Avg daily Return: ")
    print("Strategy: ", avg_daily_ret)
    print("Std daily Return: ")
    print("Strategy: ", std_daily_ret)
    print("Sharpe Ratio: ")
    print("Strategy: ", sharpe_ratio)


def plot_graph_SL(trades_portvals,trades_df):

    trades_portvals = trades_portvals/trades_portvals[0]

    plt.plot(trades_portvals, label="Strategy Learner", color="red")
    for date, value in trades_df.loc[trades_df['position'] > 0].iterrows():
        plt.axvline(x=date, color = "blue", alpha = 0.3, linestyle ='dashed')
    for date, value in trades_df.loc[trades_df['position'] < 0].iterrows():
        plt.axvline(x=date, color = "black", alpha = 0.3, linestyle ='dashed')
    plt.xlabel("Date")
    plt.ylabel("Position Values (normalized)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.title("Strategy Learner Result")
    plt.savefig("Strategy_Learner.png")
    plt.close()

def author():
    return 'tpham328'

if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		  		 			  		 			     			  	 
    testcode()