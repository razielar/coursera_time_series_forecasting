#!/usr/bin/env python
# coding: utf-8

# # Common Nonstationary to Stationary Transformations
# Feb 21st 2022

# In[2]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from IPython.display import display
from src.timeseriesFunctions import plot_time_series, chunks_statistics
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller test
from statsmodels.tsa.seasonal import seasonal_decompose
# Config:
SEED= 42
pd.options.display.float_format = '{:,.2f}'.format
sns.set_context("paper", font_scale= 1.5)
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.spines.top']= False
plotsize = (13, 5)
plt.rcParams['figure.figsize']= plotsize


# In[3]:


np.random.seed(SEED)

#data
time= np.arange(100)
stationary= np.random.normal(loc= 0, scale= 1.0, size= len(time))
trend= (time * 2.75) + stationary
seasonality= 10 + np.sin(time)*10

trend_seasonality= trend + seasonality + stationary
plot_time_series(time, trend_seasonality, title= "Trend Seasonality")


# In[4]:


adf_b4, pvalue_b4, usedlag_, nobs_, critical_values_, icbest_ = adfuller(trend_seasonality)
print("ADF: ", np.round(adf_b4, 4))
print("p-value: ", np.round(pvalue_b4, 4))


# ## Exercise 1: Remove Trend & Seasonality with Statsmodels

# In[5]:


ss_decomposition = seasonal_decompose(x=trend_seasonality, model='additive', period=6)
est_trend = ss_decomposition.trend
est_seasonal = ss_decomposition.seasonal
est_residual = ss_decomposition.resid
plot_time_series(time, est_trend, title= "Trend")
plt.show()
plot_time_series(time, est_seasonal, title= "Seasonality")
plt.show()
plot_time_series(time, est_residual, title= "Residuals")


# In[6]:


adf_after, pvalue_after, usedlag_, nobs_, critical_values_, icbest_ = adfuller(est_residual[3:-3])
print("ADF: ", np.round(adf_after,4))
print("p-value: ", np.round(pvalue_after, 6))


# ## Exercise 2: Remove chaing variance w/log transformation

# ## Exercise 3: Removing Autocrrelation with Differencing

# ## Exercise 4: Do it with example data

# In[7]:


mytime= np.arange(100)
data_path= "../course_data/"
dataset_1= np.load(data_path + "dataset_SNS_1.npy")
dataset_2= np.load(data_path + "dataset_SNS_2.npy")


# Dataset 1

# In[8]:


plot_time_series(mytime, dataset_1, title= "Dataset 1")
log_dataset1= np.log(dataset_1+25)
plt.show()
plot_time_series(mytime, log_dataset1, title= "Dataset 1 log transformation")


# In[9]:


plot_time_series(mytime, dataset_1, title= "Dataset 1")
plot_time_series(mytime, log_dataset1, title= "Dataset 1 log transformation")


# Dataset 2

# In[10]:


plot_time_series(mytime, dataset_2, title= "Dataset 2")
plt.show()
# Removing autocorrelation with differencing
dataset2_diff= dataset_2[:-1] - dataset_2[1:]
plot_time_series(mytime[:-1], dataset2_diff, title= "Dataset 2 Differenced")


# In[11]:


tmp= np.log(dataset2_diff+25)
plot_time_series(mytime[:-1], tmp, title= "Dataset 2 Differenced and Log")


# In[12]:


chunks1= np.split(log_dataset1, indices_or_sections= 10)
print("Dataset 1:")
chunks_statistics(chunks1)
chunks_2= np.split(dataset2_diff, indices_or_sections= 9)
print("")
print("Dataset 2:")
chunks_statistics(chunks_2)

chunks_3= np.split(tmp, indices_or_sections= 9)
print("")
print("Dataset 2: Diff and log")
chunks_statistics(chunks_3)


# In[13]:


pd.Series(log_dataset1).hist()
plt.show()
pd.Series(dataset2_diff).hist()
plt.show()


# Agumented Dickey-Fuller test

# In[14]:


adf_1, pvalue_1, usedlag_, nobs_, critical_values_1, icbest_ = adfuller(log_dataset1)
print("Dataset 1")
print("ADF: ", adf_1)
print("p-value:", pvalue_1)
print("crit values: ", critical_values_1)


# In[15]:


adf_2, pvalue_2, usedlag_, nobs_, critical_values_2, icbest_ = adfuller(dataset2_diff)
print("dataset_SNS_1")
print("ADF: ", adf_2)
print("p-value:", pvalue_2)
print("crit values: ", critical_values_2)


# In[16]:


adf_2, pvalue_2, usedlag_, nobs_, critical_values_2, icbest_ = adfuller(tmp)
print("dataset_SNS_1")
print("ADF: ", adf_2)
print("p-value:", pvalue_2)
print("crit values: ", critical_values_2)

