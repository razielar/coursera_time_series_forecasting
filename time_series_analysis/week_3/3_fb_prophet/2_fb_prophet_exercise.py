#!/usr/bin/env python
# coding: utf-8

# # Prophet Exercise
# March 1st 2022

# In[1]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


# In[2]:


# Custom functions:
import src.colorsetup


# # 1) Predicting C02

# In[3]:


data_path= 'course_data/'
co2= pd.read_csv(data_path+'co2-ppm-mauna-loa-19651980.csv',
                 header= 0, names= ['idx', 'co2'],
                 engine= 'python', skipfooter= 2)
co2 = co2.drop('idx', 1)

# recast co2 col to float
co2['co2'] = pd.to_numeric(co2['co2'])
co2.drop(labels=0, inplace=True)

# set index
index = pd.date_range('1/1/1965', periods=191, freq='M')
co2.index = index
co2.head()


# In[4]:


# Prophet needs as input: ds => date & y => for time-series data
co2['ds']= co2.index
co2.rename(columns= {'co2':'y'}, inplace= True)
co2.head()


# In[ ]:


model= Prophet()
model.fit(co2)


# In[6]:


# 10 years into the future
future= model.make_future_dataframe(periods= 120, freq= 'M', include_history= True)
display(co2.tail())
future.tail()


# In[7]:


# Populate forecast:
forecast= model.predict(future)
print(forecast.columns)
forecast[['ds', 'yhat']].tail()


# In[8]:


forecaset_plot= model.plot(forecast)


# In[9]:


forecast_components= model.plot_components(forecast)


# ### Additional modifications

# In[10]:


# seasonality_prior_scale: 
    # Parameter modulating the strength of the seasonality model. 
    # Larger values allow the model to fit larger seasonal fluctuations.
    # Whereas smaller values dampen (desalentar) the seasonality. 
    # Can be specified for individual seasonalities using add_seasonality.

changepoint_prior_scale=0.05 # Default value
seasonality_prior_scale= 0.00001 # Default is= 10.0
growth= 'logistic' # Default= linear
# When you're working with logistic growth you need to add a cap i.e. we limit our theoretical growth
co2['cap'] = 350
co2.head()


# In[11]:


m= Prophet(growth= growth, seasonality_prior_scale= seasonality_prior_scale, changepoint_prior_scale= changepoint_prior_scale)
m.fit(co2)
# We need to add cap to future data
future= m.make_future_dataframe(periods= 120, freq= 'M', include_history= True) # 10 years
future['cap']= 350
forecast= m.predict(future)


# In[19]:


forecast_plot= m.plot(forecast, figsize= (14,6), xlabel= "Time (months)", ylabel= "CO2")


# # 2) PM: Beijing dataset

# In[20]:


df_beijing= pd.read_csv(data_path+'FiveCitiesPM/Beijing.csv')
df_beijing.head()


# In[21]:


# Due date type we need to create our own ds
def make_date(row):
    return datetime(year= row['year'], month= row['month'], day= row['day'], hour= row['hour'])


# In[22]:


df_beijing['date']= df_beijing.apply(make_date, axis= 1)
# df_beijing.head()
df_beijing.set_index('date', inplace= True)
df_beijing['ds']= df_beijing.index
# Take only required fileds
df_beijing.head()
df= df_beijing[['ds', 'PM_Dongsi']].rename(columns= {'PM_Dongsi': 'y'})
df.tail()


# In[23]:


# Train and test sets:
df_train = df.loc['2015-11']
df_test = df.loc['2015-12':'2015-12-15']
print("Train shape: {}".format(df_train.shape))
display(df_train.tail())
print('')
print('Test shape: {}'.format(df_test.shape))
display(df_test.tail())


# In[24]:


m= Prophet()
m.fit(df_train)


# In[25]:


# 15 days
future = m.make_future_dataframe(periods = 15*24,freq = 'h') # could also leave default freq of days and do 31 for period
display(future.head())
future.tail()


# In[26]:


forecast = m.predict(future)
print(forecast.columns)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[28]:


m.plot(forecast, figsize= (17,7), xlabel= "Time", ylabel= "Air quality in Beijing")
plt.plot(df_test.y, 'r--', label= "Validation set")
plt.legend(loc= 'upper left')
plt.show()


# In[29]:


forecase_components= m.plot_components(forecast, figsize= (16,9))


# We observe the trend is very strong, so we need to change that

# In[ ]:


# Decreasted drastically from defaults
changepoint_prior_scale = 0.0005 # Default => 0.05
seasonality_prior_scale = 10.0

m= Prophet(seasonality_prior_scale= seasonality_prior_scale, changepoint_prior_scale= changepoint_prior_scale)
m.fit(df_train)
forecast= m.predict(future)


# In[ ]:


# df_cv= cross_validation(m, horizon= '365 days', period= '180 days', initial= '730 days')
df_cv= cross_validation(m, horizon= 15*24, period= '60 hours')


# In[49]:


print("CV shape: {}, Number of cutoffs: {}".format(df_cv.shape, len(df_cv.cutoff.unique())))
metrics= performance_metrics(df_cv)
metrics
# display(metrics[['horizon', 'rmse', 'mape']])
# print("Mean-RMSE: {:.4f} and Mean-MAPE: {:.2%}".format(np.mean(metrics['rmse']), np.mean(metrics['mape'])))
# print("Horizon of 365 days RMSE: {:.4f} and MAPE: {:.2%}".format(metrics.iloc[metrics.shape[0]-1]['rmse'], metrics.iloc[metrics.shape[0]-1]['mape']))


# In[42]:


image_dir= '../../../plots/time_series_analysis/'
plot_name= 'prophet.tunning.parameters.png'

# Plotting:
forecast_plot= m.plot(forecast, figsize= (17,7), xlabel= "Time", ylabel= "Air quality in Beijing")
ax = forecast_plot.gca()
ax.set_title("Model= Prophet(changepoint_prior_scale, seasonality_prior_scale)")
plt.plot(df_test.y, 'r--', label= "Validation set")
plt.legend(loc= 'upper left')
plt.show()

forecast_plot.savefig(image_dir+plot_name)


# We observe auto-correlation on the validation set.  
# **Prophet** does not do a great job of picking **auto-correlation** bc it's not additive model. Does not incorporate the relationship with past values

# In[34]:


forecast_components= m.plot_components(forecast, figsize= (15,9))


# ## Cross-validation

# In[ ]:


# future = m.make_future_dataframe(periods = 15*24,freq = 'h')
# df_cv= cross_validation(m, horizon= '365 days', period= '180 days', initial= '730 days')
df_cv= cross_validation(m, horizon= 15*24, period= '60 hours')


# In[38]:


display(df_cv)
metrics= performance_metrics(df_cv)
metrics[['rmse', 'mape']].round(4)


# In[51]:


import session_info
session_info.show()


# In[ ]:





# In[ ]:




