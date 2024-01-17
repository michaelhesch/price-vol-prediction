'''
Create a signal based trading strategy using GARCH model
Built following tutorial at: https://www.youtube.com/watch?v=A2Qge2GGmrI
'''
#%% matplotlib ipympl
import warnings
import pandas_datareader.data as web
import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from numpy.linalg import LinAlgError
from itertools import product
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

stock_df = yf.download(tickers='NVDA',
                       start='2012-01-01',
                       end='2024-01-16')

# Calculate log returns of the stock
nvda_log = np.log(stock_df['Adj Close']).diff().dropna()

# Create function to visualize stock data
# Looking for 
def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    
    x.plot(ax=axes[0][0], title='Residuals')
    x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
    
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    
    probplot(x, plot=axes[0][1])
    
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis: {kurtosis:9.2f}'

    axes[0][1].text(x=.02, 
                    y=.75, 
                    s=s,
                    transform=axes[0][1].transAxes)
    
    plot_acf(x=x, lags=lags, zero=False, auto_ylims=True, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, auto_ylims=True, ax=axes[1][1])
    
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    return

plot_correlogram(x=nvda_log,
                 lags=120,
                 title=None)
