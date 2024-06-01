#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:11:30 2024

@author: shuaitaotan
"""
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

# Fetch S&P 500 companies list and prepare the symbols list
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

# Define date range
end_date = '2024-5-25'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

# Download stock data
df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# Create SQLAlchemy engine
engine = create_engine('sqlite:///sp500_data.db')  # Using SQLite for simplicity

# Write DataFrame to SQL database
df.to_sql('sp500_prices', con=engine, if_exists='replace')

