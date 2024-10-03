""""""
"""MC2-P1: Market simulator.  		  	   		  		 			  		 			     			  	 
 		  		 			  		 			     			  	 
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def author():
    return ''


def compute_portvals(
        # orders_file="./orders/orders.csv",
        trades_df,
        start_val=100000,
        commission=9.95,
        impact=0.005,
):

    # colect orders and sort by date. orders structure Date,Symbol,Order,Shares
    # trades_df should have index as 'Date','Symbol','Order','Shares'
    trades_df = trades_df.sort_values(by='Date')
    symbols = trades_df.columns[0]
    # print(symbols)

    # build prices data frame
    # symbols = list(set(trades_df['Symbol'].tolist()))  # colect unique symbol list from orders
    start_date = trades_df.index[0]  # colect date range
    end_date = trades_df.index[-1]
    # get data to build prices
    prices = get_data([symbols], pd.date_range(start_date, end_date), addSPY=False)
    prices = prices.dropna(how='any', subset=[symbols])
    # print(prices)
    prices['Cash'] = np.ones(prices.shape[0])  # add cash column, price = 1

    # Build trades data frame to recording trades
    trades = prices * 0
    trades.fillna(0, inplace=True)

    # Add order to trades df
    for i, row in trades_df.iterrows():
        symbol = trades_df.columns[0]
        # if row[symbols] > 0:
        trades.loc[i, symbol] += row[symbols]
        trades.loc[i, 'Cash'] += - row[symbols] * prices.loc[i, symbol] * (1 + impact) - commission
        # if row[symbols] < 0:
        #     trades.loc[i, symbol] += row[symbols]
        #     trades.loc[i, 'Cash'] += row['Shares'] * prices.loc[i, symbol] * (1 - impact) - commission

    # Create position data frame to record current position
    position = trades * 0  # position have same structure as trades df
    # initiate beginning cash
    position.iloc[0, -1] = start_val
    position.iloc[0] = position.iloc[0] + trades.iloc[0]
    for i in range(1, len(trades)):
        position.iloc[i] = position.iloc[i - 1] + trades.iloc[i]

    # create portvals data frame to record position values
    prices = prices.reindex(columns=position.columns)
    portvals = position * prices
    portvals['Total'] = portvals.sum(axis=1)
    # prices.to_csv('prices.txt', sep='\t', index=True)
    # trades.to_csv('trades.txt', sep='\t', index=True)
    # position.to_csv('position.txt', sep='\t', index=True)
    # portvals.to_csv('portvals.txt', sep='\t', index=True)

    portvals = portvals.iloc[:, -1]
    # print(portvals)
    # portvals is single column of current portfolio value of every
    # single day from start to end date in trades_df, exclude non-trading days
    return portvals

def port_stats(portvals):
    daily = (portvals / portvals.shift(1)) - 1
    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily.mean()
    std_daily_ret = daily[1:].std()
    sharpe_ratio = np.sqrt(252) * np.mean(avg_daily_ret - 0) / std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    print(portvals)
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(portvals)

    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
