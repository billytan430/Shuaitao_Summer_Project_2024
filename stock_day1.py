#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')


# ## DAY 1: Generate the sp 500 datatable and create a sql dataset which updates daily at 2:00 AM

# In[2]:


sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2024-5-25'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()

df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()


# In[3]:


df.head(1000)


# ## Feature Engineering & Learning

# ### RSI, GKV, BB do not require standardization

# ## Interpretation of RSI
# ### RSI > 70: The asset is generally considered overbought and may be due for a price correction or pullback.
# ### RSI < 30: The asset is generally considered oversold and may be due for a price rebound or rally.
# ### RSI between 30 and 70: The asset is considered to be in a neutral range.

# In[4]:


### grouby(level=1)here is grouped by price ticker


# In[5]:


df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))


# ## Garman-Klass Volatility
# 
# ### Garman-Klass volatility is a valuable feature in financial analysis and modeling due to its comprehensive approach to measuring daily volatility. By incorporating high, low, open, and close prices, it provides a more accurate and reliable estimate of volatility, which can improve risk management, trading strategies, and predictive models. Integrating GK volatility into your financial workflows can enhance the robustness and accuracy of your analysis and decision-making processes.

# In[6]:


df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)


# ## Bollinger Bands
# ### Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity. They were developed by John Bollinger in the 1980s and are widely used in technical analysis to identify overbought or oversold conditions and to predict potential market reversals.
# 
# 

# In[7]:


df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                          
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                          
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])


# ## ATRAverage True Range (ATR):
# 
# ### The ATR is the moving average of the true range over a specified period, usually 14 days.
# 

# In[8]:


## transform works with only one column


# In[9]:


def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std()) ## standardize atr

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)


# ## Moving Average Convergence Divergence (MACD) indicator, the two moving averages referred to are:
# 
# ### Short-term Exponential Moving Average (EMA):
# 
# This is typically the 12-period EMA, which reacts more quickly to recent price changes due to its shorter period.
# It represents the short-term trend and highlights recent price movements more clearly.
# 
# ### Long-term Exponential Moving Average (EMA):
# 
# This is usually the 26-period EMA, which reacts more slowly to price changes because it averages over a longer period.
# It represents the long-term trend and smooths out more of the daily price fluctuations.

# In[10]:


## iloc returns the first column, which is the MACD column,and this is because the package generates a dataframe


# ## Read the MACD Score
# ## Typical Value Ranges:
# 
# ### -1 to 1: Most of the values (approximately 68% of data in a normal distribution) fall within one standard deviation of the mean. This range can be considered as the normal fluctuation around the mean.
# ### -2 to 2: Approximately 95% of the data in a normal distribution falls within two standard deviations of the mean. Values in this range are more significant but still within expected limits.
# ### Beyond -2 or 2: Values beyond two standard deviations from the mean are relatively rare (approximately 5% of the data in a normal distribution). These values indicate unusually strong movements and can be interpreted as strong bullish or bearish signals depending on the direction.

# In[11]:


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)


# ## Dollar Volume

# In[12]:


df['dollar_volume'] = (df['adj close']*df['volume'])/1e6


# In[13]:


df


# ## Monthly Level

# In[14]:


df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume')


# In[15]:


last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]


# In[16]:


df.unstack()[last_cols].resample('M').last().stack('ticker')


# In[17]:


data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                         df.unstack()[last_cols].resample('M').last().stack('ticker')],axis=1)).dropna()
data


# In[18]:


data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
data


# In[19]:


data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

data


# In[20]:


### df.xs is a powerful method for extracting specific cross-sections of data from MultiIndex DataFrames


# ## Monthly Returns

# In[21]:


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
    
    
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

data


# ## Fama French Factor

# In[22]:


factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()

factor_data


# In[23]:


observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

factor_data


# In[24]:


betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

betas 


# In[25]:


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop('adj close', axis=1)

data = data.dropna()

data.info()


# ## K Mean

# In[26]:


target_rsi_values = [30, 45, 55, 70]

initial_centroids = np.zeros((len(target_rsi_values), 18))

initial_centroids[:, 0] = target_rsi_values

initial_centroids


# ### If the initialization of the cluster is random, when we produce the plots, the cluster colors and labels will be all over the places, however, we want to classify the rsi over 70 into one cluster therefore we define them/

# In[31]:


from sklearn.cluster import KMeans

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

data


# In[32]:


def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,6] , cluster_0.iloc[:,0] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,6] , cluster_1.iloc[:,0] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,6] , cluster_2.iloc[:,0] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,6] , cluster_3.iloc[:,0] , color = 'black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return


# In[33]:


plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    
    g = data.xs(i, level=0)
    
    plt.title(f'Date {i}')
    
    plot_clusters(g)


# ### CLUSTER 3 Upward Momentum

# In[34]:


## Based On Cluster 3, How to Form a Portfolio


# In[35]:


filtered_df = data[data['cluster']==3].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(1)
## Beginning of the next month

filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    
fixed_dates


# ## Portfolio Maximization

# In[178]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[179]:


## Pacakges may have version errors. Establish a New Environment can safely resolve the issue.


# In[204]:


def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .25),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()


# In[205]:


stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

new_df


# In[207]:


returns_dataframe = np.log(new_df['Adj Close']).diff()

for start_date in fixed_dates.keys():
    
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]

        try:
            weights = optimize_weights(prices=optimization_df, lower_bound=round(1/(len(optimization_df.columns)*2), 4))
            weights = pd.DataFrame(weights, index=pd.Series(0))
            success = True

        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
            weights = pd.DataFrame([1/len(optimization_df.columns) for _ in range(len(optimization_df.columns))],
                                   index=optimization_df.columns.tolist(),
                                   columns=pd.Series([0])).T
            success = False

        temp_df = returns_dataframe[start_date:end_date]
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0) \
            .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                   left_index=True,
                   right_index=True) \
            .reset_index().set_index(['Date', 'Ticker']).unstack().stack()

        temp_df.index.names = ['date', 'ticker']
        temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

    except Exception as e:
        print(e)

portfolio_df = portfolio_df.drop_duplicates()
portfolio_df


# ## Comparison & Visualization 

# In[209]:


import matplotlib.ticker as mtick
import pandas as pd

plt.style.use('ggplot')

portfolio_df = portfolio_df.sort_index()  # Sort the DataFrame by the DatetimeIndex

portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

nearest_date = pd.date_range(end='2023-09-29', periods=1, freq='D')[0]
portfolio_cumulative_return[:nearest_date].plot(figsize=(16,6))

plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()


# In[ ]:




