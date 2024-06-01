#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:56:15 2024

@author: shuaitaotan
"""
import numpy as np
import sqlite3
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas_datareader.data as web
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Define your database path
db_path = '/Users/shuaitaotan/Documents/GitHub/Shuaitao_Summer_Project_2024/sp500_data.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Read S&P 500 company symbols from the Wikipedia page
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

# Define the date range
end_date = '2024-05-25'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

# Download the stock data
df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# Feature Engineering
# RSI
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.rsi(close=x, length=20))

# Garman-Klass Volatility
def garman_klass_volatility(data):
    log_hl = (data['high'] / data['low']).apply(np.log)
    log_co = (data['close'] / data['open']).apply(np.log)
    rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return rs.rolling(window=10).sum().apply(np.sqrt) * np.sqrt(252 / 10)

df['gkv'] = df.groupby(level=1).apply(garman_klass_volatility).reset_index(level=0, drop=True)

# Bollinger Bands
def calculate_bbands(group):
    bb = ta.bbands(close=group['adj close'], length=20, std=2, mamode='sma')
    return bb['BBU_20_2.0'], bb['BBM_20_2.0'], bb['BBL_20_2.0']

bbands = df.groupby(level=1).apply(calculate_bbands)
df['bb_upper'] = bbands.apply(lambda x: x[0])
df['bb_middle'] = bbands.apply(lambda x: x[1])
df['bb_lower'] = bbands.apply(lambda x: x[2])

# Momentum
df['momentum'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.mom(close=x, length=10))

# SMA
df['sma'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.sma(close=x, length=20))

# WMA
df['wma'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.wma(close=x, length=30))

# Z-Score
df['zscore'] = df.groupby(level=1)['adj close'].transform(lambda x: (x - x.rolling(window=20).mean()) / x.rolling(window=20).std())

# Calculate Returns
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    return df

df = df.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# Fama French Factor
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data = factor_data.join(df['return_1m']).sort_index()

observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

betas = (factor_data.groupby(level=1, group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))

factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
df = df.join(betas.groupby('ticker').shift())
df.loc[:, factors] = df.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

df = df.drop('adj close', axis=1)
df = df.dropna()

# Reset index for saving to SQL
df.reset_index(inplace=True)

# Create the new table `sp500_features` and insert data
df.to_sql('sp500_features', conn, if_exists='replace', index=False)

# Close the database connection
conn.close()

print("Features have been successfully added to the sp500_features table.")