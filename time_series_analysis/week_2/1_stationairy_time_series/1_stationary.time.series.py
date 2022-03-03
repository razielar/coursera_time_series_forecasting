#!/usr/bin/env python
# coding: utf-8

# # Stationary Time-series
# Feb 21st 2022

# In[1]:


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
# Config:
pd.options.display.float_format = '{:,.2f}'.format
sns.set_context("paper", font_scale= 1.5)
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.spines.top']= False
SEED= 42
plotsize = (13, 5)
plt.rcParams['figure.figsize']= plotsize


# ## Generate time-series data

# In[2]:


np.random.seed(SEED)

#data
time= np.arange(100)
stationary= np.random.normal(loc= 0, scale= 1.0, size= len(time))


# In[3]:


def plot_time_series(x,y, title= "Title", xlabel= "time", ylabel= "series"):
    plt.plot(x,y,'k-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha= 0.3)


# In[4]:


plot_time_series(x= time, y= stationary, title= "Stationary Time-series")


# In[5]:


# Generate a time-series toy example with high-autocorrelation
seed= 3.14

lagged= np.empty_like(time, dtype= 'float')
for i in time:
    lagged[i]= seed + np.random.normal(loc= 0, scale= 2.5, size=1)
    seed= lagged[i]
plot_time_series(time, lagged, title= "Non-stationary Time-series")    


# ### Non-stationary Time-series
# * 1) Trend
# * 2) Heteroscedasticity
# * 3) Seasonality
# * 4) Trend + seasonality

# In[6]:


# Trend
trend= (time * 2.75) + stationary
plot_time_series(time, trend, title= "Non-stationarity Time-series w/Trend")


# In[7]:


# Heteroscedasticity
seed_2= 1234
np.random.seed(seed_2)

# data
level_1= np.random.normal(loc= 0, scale= 1.0, size= 50)
level_2= np.random.normal(loc= 0, scale= 10.0, size= 50)
heteroscedasticity = np.append(level_1, level_2)

plot_time_series(time, heteroscedasticity, title= "Non-stationary Time-series w/heteroscedasticity")


# In[8]:


# Seasonality
seasonality= 10 + np.sin(time)*10
plot_time_series(time, seasonality, title= "Non-stationary Time-series w/seasonality")


# In[9]:


# Trend and seasonality
trend_seasonality= trend + seasonality + stationary
plot_time_series(time, trend_seasonality, title= "Non-stationary Time-series w/Trend + Seasonality")


# ## Exercises

# In[12]:


mytime= np.arange(100)
data_path= "../course_data/"
dataset_1= np.load(data_path + "dataset_SNS_1.npy")
dataset_2= np.load(data_path + "dataset_SNS_2.npy")


# In[13]:


plot_time_series(mytime, dataset_1, title= "Dataset 1")


# In[14]:


plot_time_series(mytime, dataset_2, title= "Dataset 2")


# In[ ]:




