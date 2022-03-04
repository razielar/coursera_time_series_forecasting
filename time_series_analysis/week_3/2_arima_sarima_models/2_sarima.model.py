#!/usr/bin/env python
# coding: utf-8

# # SARIMA mode
# Feb 24th 2022

# In[1]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import warnings
warnings.filterwarnings('ignore')
import pmdarima as pm #help us to select p,d,q
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error


# In[2]:


# Custom functions:
from src.timeseriesFunctions import get_years
import src.colorsetup


# ## Data Preparation and EDA

# In[3]:


data= "./course_data/"
monthly_temp= pd.read_csv(data+"mean-monthly-temperature-1907-19.csv", 
                          names= ["month", "temp"], skipfooter= 2, engine= 'python',
                          infer_datetime_format= True, header= 0, index_col= 0)
monthly_temp.index= pd.to_datetime(monthly_temp.index)
# 'A'= annually
annual_temp=  monthly_temp.resample('A').mean()
monthly_temp


# In[4]:


monthly_temp.describe()


# In[5]:


get_years(monthly_temp)


# In[6]:


# Plot the temperature per year
fig, axes = plt.subplots(2,1, figsize= (18, 8))

axes[0].plot(monthly_temp)
axes[0].set_title("Monthly data")
axes[1].plot(annual_temp)
axes[1].set_title("Annual data")

plt.show()


# In[7]:


plt.plot(monthly_temp, label= "Monthly")
plt.plot(annual_temp, label= "Annual")
plt.legend()
plt.grid()
plt.show()


# In[8]:


image_folder= "../../../plots/time_series_analysis/"
plot_name= "seasonal.analysis.png"

violin_plot= sns.violinplot(x= monthly_temp.index.month, y= monthly_temp.temp)
violin_plot.set_xlabel("Months")
violin_plot.set_ylabel("Tempeture")
violin_plot.set_title("Simple seasonal analysis")
fig= violin_plot.get_figure()
fig.savefig(image_folder+plot_name, transparent= True, bbox_inches= 'tight')


# In[9]:


# Differencing by season:
monthly_temp['lag_12']= monthly_temp.shift(12)
display(monthly_temp.head(14))
monthly_temp['seasonal_diff']= monthly_temp.temp - monthly_temp.lag_12
display(monthly_temp.head(14))
# Log transformation for differencing by season
monthly_temp['seasonal_diff_log']= np.log10(np.array(monthly_temp.seasonal_diff) + monthly_temp.seasonal_diff.max())
display(monthly_temp.head(14))


# In[10]:


plot_name= "seasonal.differencing.png"

fig, axes = plt.subplots(2,1, figsize= (17,8), sharex= True)

axes[0].plot(monthly_temp.temp)
axes[0].set_title("Original data")

axes[1].plot(monthly_temp.seasonal_diff)
axes[1].set_title("Seasonal Differencing")

# axes[2].plot(monthly_temp.seasonal_diff_log)
# axes[2].set_title("Log10(Seasonal Differencing+25)")

plt.tight_layout()
plt.savefig(image_folder+plot_name, transparent= True)
plt.show()


# In[11]:


def dickey_fuller_test(timeseries):
    """
    NUll hypothesis= time-series is not stationary
    """
    test= adfuller(timeseries)
    df= pd.Series(test[0:4], index= ["Test-statistics", "p-value", "Lags-used", "Observation-used"])
    for key,value in test[4].items():
        df['Critical value (%s)'%key] = value    
    print(df)
    # Rolling
    rolmean= timeseries.rolling(window=12).mean()
    rolstd= timeseries.rolling(window=12).std()
    # Plotting
    original= plt.plot(timeseries, color= "blue", label= "Original Time-series")
    mean= plt.plot(rolmean, color= "red", label= "Rolling Mean")
    std= plt.plot(rolstd, color= "black", label= "Rolling Std")
    plt.legend(loc= "best")
    plt.grid()
    plt.show()


# In[12]:


dickey_fuller_test(monthly_temp.temp)


# In[13]:


dickey_fuller_test(annual_temp.temp)


# ## SARIMA with Statsmodels

# In[14]:


def eda_plots(data, lags= None, save= False, save_path= ''):
    plt.rcParams['figure.figsize']= (22,5)
    layout= (1,3)
    raw= plt.subplot2grid(layout, (0,0))
    pacf= plt.subplot2grid(layout, (0,1))
    acf= plt.subplot2grid(layout, (0,2))
    
    raw.plot(data)
    plot_acf(data, lags= lags, ax= acf, zero= False)
    plot_pacf(data, lags= lags, ax= pacf, zero= False)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, transparent= True)

plot_name= 'eda.pacf.acf.png'        
eda_plots(monthly_temp.temp, lags= 36, save= True, save_path= image_folder+plot_name)


# ACF Shape|Indicated Model
# ---|---
# Exponential, decaying to zero|Autoregressive model. Use the partial autocorrelation plot to identify the order of the autoregressive model.
# Alternating positive and negative, decaying to zero|Autoregressive model. Use the partial autocorrelation plot to help identify the order.
# One or more spikes, rest are essentially zero|Moving average model, order identified by where plot becomes zero.
# Decay, starting after a few lags|Mixed autoregressive and moving average (ARMA) model.
# All zero or close to zero|Data are essentially random.
# High values at fixed intervals|Include seasonal autoregressive term.
# No decay to zero|Series is not stationary.

# In[15]:


plot_name= "eda.differencing.pacf.acf.png"
eda_plots(monthly_temp.seasonal_diff.dropna(), lags= 36, save= True, save_path= image_folder+plot_name)


# In[16]:


# 1) seasonal_order= P,D,Q,s; s= periodicity. s= 12 ==> monthly data 
# 2) trend= 'c'; means constant trend
sar= SARIMAX(monthly_temp.temp, order= (1,0,0), seasonal_order= (0,1,1,12), trend= 'c').fit()


# In[17]:


# A|C and B|C, we want this values as low as possible. Measure the complexity of the model
sar.summary()


# In[18]:


# Plot the residuals
plot_name= "eda.residuals.pacf.acf.png"
eda_plots(sar.resid[sar.loglikelihood_burn:], lags= 12, save= True, save_path= image_folder+plot_name)


# In[19]:


diagnostics= sar.plot_diagnostics(lags= 12, figsize= (20,10))


# In[20]:


monthly_temp['forecast']= sar.predict(start= 750, end= 792)
# display(monthly_temp[['temp', 'forecast']])
# Plot
monthly_temp[750:][['temp', 'forecast']].plot()
plt.grid()
plt.show()


# In[21]:


sar2= SARIMAX(monthly_temp.temp, order=(3,0,0), seasonal_order=(0,1,1,12), trend='c').fit()
sar2.summary()


# In[22]:


monthly_temp['forecast_2'] = sar2.predict(start = 750, end= 790, dynamic=False)  
plt.plot(monthly_temp[730:][['temp', 'forecast_2']])
plt.grid()


# Future forecast

# In[23]:


future_forecast= sar2.get_forecast(50)
confidence_int= future_forecast.conf_int(alpha= 0.01)
# Pull predicted mean
fcst= future_forecast.predicted_mean
# Plot
plt.plot(monthly_temp.temp[-50:])
plt.plot(fcst)
plt.fill_between(confidence_int.index, confidence_int['lower temp'], confidence_int['upper temp'], alpha= 0.3)
plt.grid()
plt.show()


# ### Statistical tests:
# * 1) Jarque-Bera => Normality. Null hypothesis= residuals are normally distributed
# * 2) Ljung-Box => Serial correlaiton. Null hypothesis= is no serial correlation in residuals (independent of each other)
# * 3) Heteroskedasticity => change in variance for residuals. Null hypothesis= is not heteroskedasticity
# * 4) Durbin-Watson => Test autocorrelation of residuals. We want 1-3, 2 is ideal (no serial correlation)

# In[24]:


norm_val, norm_p, skew, kurtosis = sar.test_normality('jarquebera')[0]
lb_val, lb_p = sar.test_serial_correlation(method='ljungbox',)[0]
het_val, het_p = sar.test_heteroskedasticity('breakvar')[0]

# we want to look at largest lag for Ljung-Box, so take largest number in series
lb_val = lb_val[-1]
lb_p = lb_p[-1]

durbin_watson = sm.stats.stattools.durbin_watson(
    sar.filter_results.standardized_forecasts_error[0, sar.loglikelihood_burn:])

print('Normality: val={:.3f}, p={:.3f}'.format(norm_val, norm_p))
print('Ljung-Box: val={:.3f}, p={:.3f}'.format(lb_val, lb_p))
print('Heteroskedasticity: val={:.3f}, p={:.3f}'.format(het_val, het_p))
print('Durbin-Watson: d={:.2f}'.format(durbin_watson))


# ## Autofit models

# In[25]:


get_ipython().run_cell_magic('time', '', '# m ==> The period for seasonal differencing, ``m`` refers to the number of periods in each season. For example, ``m`` is 12 for monthly data\nstepwise_model = pm.auto_arima(monthly_temp.temp, \n                               start_p=0, max_p= 3,\n                               d= 0, \n                               start_q=0, max_q= 3,\n                               m=12, seasonal=True,\n                               start_P=0, \n                               D=1, \n                               start_Q= 0,\n                               trace=True, error_action=\'ignore\', suppress_warnings=True, stepwise=True)\nprint(\'\')\nprint(\'-\'*28)\nprint("A|C: {:.3f}\\np,d,q: {}\\nP,D,Q: {}".format(stepwise_model.aic(), stepwise_model.order, stepwise_model.seasonal_order)) ')


# In[26]:


# diagnostics= sar.plot_diagnostics(lags= 12, figsize= (20,10))
stepwise_diag= stepwise_model.plot_diagnostics(lags=12, figsize= (20,10))


# In[27]:


def future_preds_df(model,series,num_months):
    """
    Generate a df with model predictions
    """
    pred_first = series.index.max()+relativedelta(months=1)
    pred_last = series.index.max()+relativedelta(months=num_months)
    date_range_index = pd.date_range(pred_first,pred_last,freq = 'MS')
    vals = model.predict(n_periods = num_months)
    return pd.DataFrame(vals,index = date_range_index)


# In[32]:


predictions= future_preds_df(stepwise_model, monthly_temp.temp, 50)

plot_name= 'sarima.predictions.png'
plt.plot(monthly_temp.temp[-300:], label= "Time-series data")
plt.plot(predictions, label= "Predictions")

plt.legend()
plt.title("Auto SARIMA model: (p=1,d=0,q=1)(P=2,D=1,Q=0,s=12)")
plt.savefig(image_folder+plot_name, transparent= True)
plt.show()


# In[33]:


import session_info
session_info.show()


# In[ ]:





# In[ ]:




