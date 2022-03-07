#!/usr/bin/env python
# coding: utf-8

# # Forecasting exercise for SARIMA
# Feb 28th 2022

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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as mape
import pmdarima as pm #help us to select p,d,q
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Custom functions:
import src.colorsetup
from src.timeseriesFunctions import plot_time_series, get_years, dickey_fuller_test, mape, cross_validate 


# ## Input data

# In[3]:


data_path = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv'
data= pd.read_csv(data_path, usecols= ['time', 'value'], index_col= 'time', parse_dates= ['time'])
data


# In[4]:


plot_time_series(data.index, data.value, title= "Yearly Sunpots", ylabel= "Sunspots")


# In[5]:


get_years(data)


# In[6]:


cutoff_1= [str(i) for i in range(1700,1800)]
cutoff_2= [str(i) for i in range(1800,1900)]
cutoff_3= [str(i) for i in range(1900,1988)]

print(np.std(data.loc[cutoff_1]['value']))
print(np.std(data.loc[cutoff_2]['value']))
print(np.std(data.loc[cutoff_3]['value']))


# In[7]:


# Return the natural logarithm of one plus the input array, element-wise
data['log_transformation']= np.log1p(data.value)
print(np.std(data.loc[cutoff_1]['log_transformation']))
print(np.std(data.loc[cutoff_2]['log_transformation']))
print(np.std(data.loc[cutoff_3]['log_transformation']))


# In[8]:


plot_time_series(data.index, data.log_transformation, title= "Yearly Sunpots", ylabel= "log(Sunspots)")


# In[9]:


seasonal_plot= seasonal_decompose(data.log_transformation, period= 11).plot()


# In[10]:


acf= plot_acf(data.log_transformation, zero= False)


# In[11]:


pacf= plot_pacf(data.log_transformation, zero= False)


# Differencing

# In[12]:


data['lag_11']= data.log_transformation.shift(11)
data['seasonal_diff'] = data.log_transformation - data['lag_11']


# In[ ]:


# data.head(12)


# In[13]:


dickey_fuller_test(data['seasonal_diff'].dropna())


# In[14]:


acf_seasonal_diff= plot_acf(data['seasonal_diff'].dropna(), zero= False)
pacf_seasonal_diff= plot_pacf(data['seasonal_diff'].dropna(), zero= False)


# ## Start with a very simple model

# In[15]:


start_model= SARIMAX(data.log_transformation, order= (0,0,0), seasonal_order= (0,1,0,12), trend= 'c').fit()


# In[17]:


pacf_residuals= plot_pacf(start_model.resid[start_model.loglikelihood_burn:], zero= False)


# In[18]:


auto_model= pm.auto_arima(data.log_transformation,
                          start_p= 0, max_p= 3, 
                          d= 0,
                          start_q= 0, max_q= 3,
                          m= 11, 
                          seasonal= True, 
                          start_P= 0,
                          D= 1, 
                          trace= True, error_action= 'ignore', suppress_warnings= True, stepwise= True)


# In[19]:


print('Order: {} Seasonal order: {}'.format(auto_model.order, auto_model.seasonal_order))


# In[ ]:


eda_model= SARIMAX(data.log_transformation, order= (2,0,0), seasonal_order= (2,1,0,11), trend= 'c').fit()


# In[21]:


auto_diag= eda_model.plot_diagnostics(figsize= (18,8))


# In[ ]:


warnings.filterwarnings("ignore")
series = data['log_transformation']
horizon = 3
start = int(len(data.value)*.75) # 75% for training_set
step_size = 1
order = auto_model.order
seasonal_order = auto_model.seasonal_order

log_cv1 = cross_validate(series,horizon,start,step_size,
                    order = order,
                    seasonal_order = seasonal_order)


# In[23]:


exp_cv1= np.expm1(log_cv1)


# In[24]:


display(data.tail())
exp_cv1.tail()


# In[33]:


image_folder= "../../../plots/time_series_analysis/"
plot_name= "sarima.forecast.png"


print("MAPE: {:.4f}".format(mape(exp_cv1)))
exp_cv1.actual.plot(color= "black")
exp_cv1.forecast.plot(color= "orange")

plt.title("SARIMA model with MAPE: {:.4f}".format(mape(exp_cv1)))
plt.ylabel("Time series")
plt.xlabel("Time (Years)")
plt.legend()
plt.grid(alpha= 0.4)
plt.savefig(image_folder+plot_name, transparent= True)
plt.show()


# In[34]:


# Zoom in
exp_cv1['actual'].loc['1970':'1988'].plot(color= "black")
exp_cv1['forecast'].loc['1970':'1988'].plot(color= "orange") 

plt.legend()
plt.show()


# In[35]:


import session_info
session_info.show()


# In[ ]:




