#!/usr/bin/env python
# coding: utf-8

# # Identifying Stationarity
# Feb 21st 2022

# In[4]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from src.timeseriesFunctions import plot_time_series
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller test
SEED= 42


# In[3]:


# Custom functions
import src.colorsetup


# ## Identifying Stationarity
# * 1) Exploratory Data Analysis (EDA) of the time-series
# * 2) Summary statistics for mean and variance
# * 3) Agumented Dickey-Fuller test

# In[5]:


np.random.seed(SEED)

#data
time= np.arange(100)
stationary= np.random.normal(loc= 0, scale= 1.0, size= len(time))
trend= (time * 2.75) + stationary


# ### 1) EDA of the time-series

# In[6]:


# 1) EDA of the time-series
plot_time_series(time, trend, title= "Non-stationarity Time-series w/Trend")


# ### 2) Summary Statistics & Plots

# In[7]:


# split data into 10 chunks
chunks= np.split(trend, indices_or_sections= 10)
chunks[0]


# In[8]:


print("{} | {:7} | {}".format("Chunk", "Mean", "Variance"))
print("-" * 26)
for i,j in enumerate(chunks, 1):
    print("{:5} | {:.6} | {:.6}".format(i, np.mean(j), np.var(j)))


# In[9]:


def chunks_statistics(np_chunk_array):
    print("{} | {:7} | {}".format("Chunk", "Mean", "Variance"))
    print("-" * 26)
    for i,j in enumerate(np_chunk_array, 1):
        print("{:5} | {:.6} | {:.6}".format(i, np.mean(j), np.var(j)))


# In[10]:


pd.Series(trend).hist()
plt.show()


# In[11]:


pd.Series(stationary).hist()
plt.show()


# ### 3) Statistical test
# Null hypothesis= time-series is non-stationary  
# Whether or not there's a high correlation between the current value and some lag values. That will be the case if: trends

# In[12]:


adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(trend)
# adf: test statistic
print("Test statistics: {:.6}, P.value: {:.6}".format(adf, pvalue))


# ## Exercise

# In[13]:


mytime= np.arange(100)
data_path= "../course_data/"
dataset_1= np.load(data_path + "dataset_SNS_1.npy")
dataset_2= np.load(data_path + "dataset_SNS_2.npy")


# In[14]:


plot_time_series(mytime, dataset_1, title= "Dataset 1")


# In[15]:


plot_time_series(mytime, dataset_2, title= "Dataset 2")


# In[16]:


chunks_1= np.split(dataset_1, indices_or_sections= 10)
chunk_2= np.split(dataset_2, indices_or_sections= 10)

print('Dataset 1')
chunks_statistics(chunks_1)
print('')
print('Dataset 2')
chunks_statistics(chunk_2)


# In[17]:


series_1= pd.Series(dataset_1)
series_2= pd.Series(dataset_2)

fig, axes= plt.subplots(1,2)
axes[0].hist(series_1)
axes[0].set_title("Dataset 1")

axes[1].hist(series_2)
axes[1].set_title("Dataset 2")

plt.show()


# In[18]:


adf_1, pvalue_1, usedlag_, nobs_, critical_values_1, icbest_ = adfuller(dataset_1)
print("Dataset 1")
print("Augmented Dickey-Fuller: ", np.round(adf_1, 4))
print("p-value:", np.round(pvalue_1, 4))
print("crit values: ", critical_values_1)

print()

adf_2, pvalue_2, usedlag_, nobs_, critical_values_2, icbest_ = adfuller(dataset_2)
print("Dataset 2")
print("ADF: ", np.round(adf_2, 4))
print("p-value:", np.round(pvalue_2,4))
print("crit values: ", critical_values_2)


# In[19]:


import session_info
session_info.show()


# In[ ]:





# In[ ]:




