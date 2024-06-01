#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:57:18 2024

@author: shuaitaotan
"""
import pandas_datareader.data as web
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import datetime as dt
import pandas_ta
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# Fetch S&P 500 companies list and prepare the symbols list
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

# Define date range
end_date = dt.datetime.now().strftime('%Y-%m-%d')
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

# Download stock data
df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# Create SQLAlchemy engine
engine = create_engine('sqlite:///sp500_data.db')  # Using SQLite for simplicity

# Write DataFrame to SQL database (update daily)
df.to_sql('sp500_prices', con=engine, if_exists='replace')

# Feature Engineering & Learning
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])
df['atr'] = df.groupby(level=1, group_keys=False).apply(lambda stock_data: pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14).sub(pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14).mean()).div(pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14).std()))
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(lambda close: pandas_ta.macd(close=close, length=20).iloc[:,0].sub(pandas_ta.macd(close=close, length=20).iloc[:,0].mean()).div(pandas_ta.macd(close=close, length=20).iloc[:,0].std()))
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

# Monthly Level
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[[c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]].resample('M').last().stack('ticker')],axis=1)).dropna()
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
data = data.groupby(level=1, group_keys=False).apply(lambda df: df.assign(**{f'return_{lag}m': (df['adj close'].pct_change(lag).pipe(lambda x: x.clip(lower=x.quantile(0.005), upper=x.quantile(0.995))).add(1).pow(1/lag).sub(1)) for lag in [1, 2, 3, 6, 9, 12]})).dropna()

# Fama-French Factor
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data = factor_data.join(data['return_1m']).sort_index()

observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

betas = (factor_data.groupby(level=1, group_keys=False).apply(lambda x: RollingOLS(endog=x['return_1m'], exog=sm.add_constant(x.drop('return_1m', axis=1)), window=min(24, x.shape[0]), min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const', axis=1)))

factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas.groupby('ticker').shift()))
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()

# Write 'data' DataFrame to SQL database with table name 'features'
data.to_sql('features', con=engine, if_exists='replace')
