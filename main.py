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
from arch import arch_model

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

stock_df = yf.download(tickers='NVDA',
                       start='2012-01-01',
                       end='2024-01-16')

# Calculate log returns of the stock
nvda_log = np.log(stock_df['Adj Close']).diff().dropna()

# Create function to visualize stock data
# Looking for what lags in autocorrelation are most correlated
# This will help inform the model to predict price by log returns
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

# Visualize based on logs
# Log results show autocorrelation appears random
plot_correlogram(x=nvda_log,
                 lags=120,
                 title=None)

# Turn log returns into daily volitility
nvda_daily_vol = (nvda_log*nvda_log.mean())**2

# Visualize based on daily volitility
# Now we see more correlation trend using vol approach
# This will allow a model to reasonably predict next day vol
plot_correlogram(x=nvda_daily_vol,
                 lags=120,
                 title=None)

'''
Build predictive model
'''
# Use 10 years of data, 252 trading days/yr
trainsize = 10*252
# Remove outliers from the data which will skew model
# If upper or lower is higher than specified quantile, assign the quantile as value
data = nvda_log.clip(lower=nvda_log.quantile(.05),
                     upper=nvda_log.quantile(.95))
# Length of entire sample
T = len(nvda_log)

results = {}

# Loop through the combinations of autoregression model and moving avg and fit models
for p in range(1, 5):
    for q in range(1, 5):
        print(f'{p} | {q}')
        
        result = []
        for s, t in enumerate(range(trainsize, T-1)):
            train_set = data.iloc[s:t]
            test_set = data.iloc[t+1]
            model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
            forecast = model.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            result.append([(test_set-mu)**2, var])
        df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
        results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))

# Display model results, actual and predicted var
result_df = pd.DataFrame(results, index=pd.Series(0)).unstack()
print(result_df)

# Select the result with the best fit
result_df = result_df.idxmin()
print(result_df)

# run arch model with best p and q model result from step above
model_data = nvda_log.clip(lower=nvda_log.quantile(.05),
                     upper=nvda_log.quantile(.95))
best_model = arch_model(y=model_data,
                        p=4,
                        q=2).fit(update_freq=5, disp='off')
print(best_model.summary())

'''
Rolling prediction - each day fit model and predict next day var
Compare current day variance with next day prediction
Form trading strategy based on this approach
'''
# Use log data

# Calculate rolling var

