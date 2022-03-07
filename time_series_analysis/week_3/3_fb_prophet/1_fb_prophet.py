#!/usr/bin/env python
# coding: utf-8

# # Facebook Prophet
# March 1st 2022

# Facebook Prophet is an **general additive model** that includes a number of highly advanced, intelligent forecasting methods, incluiding changepoint analysis.  
# _y = g(t) + s(t) + h(t) + $\epsilon_t$_  
# Here g(t) => the **trend** function which models non-periodic changes, s(t) => **periodic** changes (e.g. weekly and yearly seasonality), and h(t) => represents the effects of **holidays** (which occur on potentially irregular scheadules over one or more days)

# Prophet was optimized with the **business forecast tasks**, typically following the next features:
# * We need a bit of data, at least a year. With hourly, daily, or weekly observations
# * Strong multiple human scale seasonality (e.g. week day)
# * Important holidays that occur at irregular intervals 
# * A reasonable number of missing observations or large outliers
# * Historical changes may not persist and die out
# * Trends that are non-linear growth curves

# In[133]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


# In[134]:


# Custom functions:
import src.colorsetup


# ## Input data
# Peyton Manning (football player) data 

# In[135]:


data_path = 'https://raw.githubusercontent.com/PinkWink/DataScience/master/data/07.%20example_wp_peyton_manning.csv'
peyton = pd.read_csv(data_path)
print("Data shape: {}".format(peyton.shape))
peyton.head()


# In[136]:


peyton.plot()
plt.title("Wikipedia searches for Peyton")
plt.grid()
plt.show()


# In[137]:


# Let's do a log transformation
peyton['y']= np.log(peyton['y'])
display(peyton.head())
peyton.ds.max()


# In[138]:


peyton.plot()
plt.grid()
plt.show()


# # 1) Fitting a vanilla model

# In[139]:


m= Prophet()
m.fit(peyton)


# In[140]:


# Forecast 365 days into future
# prophet requires a blank df to input predictions
future= m.make_future_dataframe(periods= 365)
# display(peyton)
# display(future.head())
# display(future.tail())


# In[141]:


# Populate forecast
forecast= m.predict(future)
display(forecast.head())
print(forecast.columns)


# In[142]:


display(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
forecast[forecast.ds >= '2016-01-20'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# ## Cross validate

# In[ ]:


# Period specifiy the cutoffs, every 180 days
df_cv= cross_validation(m, horizon= '365 days', period= '180 days', initial= '730 days')


# In[144]:


display(peyton.head())
df_cv.head()


# In[145]:


# We predicted a year into the future
first_cut= df_cv[df_cv.cutoff == datetime(2010,2,15)]
last_cut= df_cv[df_cv.cutoff == datetime(2015,1,20)]
display(first_cut.tail())
print("CV shape: {}, First cutoff: {}, Number of cutoffs: {}".format(df_cv.shape, first_cut.shape, len(df_cv.cutoff.unique())))


# In[189]:


metrics= performance_metrics(df_cv)
display(metrics[['horizon', 'rmse', 'mape']])
print("Mean-RMSE: {:.4f} and Mean-MAPE: {:.2%}".format(np.mean(metrics['rmse']), np.mean(metrics['mape'])))
print("Horizon of 365 days RMSE: {:.4f} and MAPE: {:.2%}".format(metrics.iloc[metrics.shape[0]-1]['rmse'], metrics.iloc[metrics.shape[0]-1]['mape']))


# In[202]:


h_1_mape= metrics.iloc[metrics.shape[0]-1]['mape']
h_1_rmse= metrics.iloc[metrics.shape[0]-1]['rmse']

image_dir= '../../../plots/time_series_analysis/'
plot_name= 'initial.prophet.model.png'

forecast_plot= m.plot(forecast, figsize= (17,7), xlabel= "Time", ylabel= "Wikipedia searches for Peyton (log)")
ax = forecast_plot.gca()
ax.set_title("Initial model= Prophet(), MAPE: {:.2%} and RMSE: {:.3f}".format(h_1_mape, h_1_rmse))
ax.set_xlabel("Time")
ax.set_ylabel("Wikipedia searches for Peyton (log)")

forecast_plot.savefig(image_dir+plot_name)


# In[190]:


plot_name= 'initial.prophet.model.cross.validation.png'

fig, axes= plt.subplots(1,2, sharey= True)
fig.set_figwidth(20)

axes[0].plot(first_cut.ds, first_cut.y, color= "black", label= "True")
axes[0].plot(first_cut.ds, first_cut.yhat, color= "orange", label= "Predicted")
axes[0].fill_between(first_cut.ds, first_cut.yhat_lower, first_cut.yhat_upper, color= "orange", alpha= 0.2)
axes[0].set_title("First fold")
axes[0].legend(loc= "upper left")

axes[1].plot(last_cut.ds, last_cut.y, color= "black", label= "True")
axes[1].plot(last_cut.ds, last_cut.yhat, color= "orange", label= "Predicted")
axes[1].fill_between(last_cut.ds, last_cut.yhat_lower, last_cut.yhat_upper, color= "orange", alpha= 0.2)
axes[1].set_title("Last fold (11th)")
axes[1].legend(loc= "upper left")

axes[0].grid(alpha= 0.3)
axes[1].grid(alpha= 0.3)

plt.tight_layout()
fig.savefig(image_dir+plot_name, transparent= True)


# In[191]:


forecast_components= m.plot_components(forecast)


# # 2) Add Holidays information
# In our case playoffs and superbowls dates

# In[192]:


playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0, # these help us specify spillover into previous and future days which will be treated as own holidays
  'upper_window': 1,
})

# display(playoffs)

superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})

# display(superbowls) 
holidays= pd.concat((playoffs, superbowls))
holidays.head()


# In[193]:


# fit and predict
m_holidays= Prophet(holidays= holidays)
forecast_holidays = m_holidays.fit(peyton).predict(future)


# In[194]:


forecast_holidays.columns


# In[195]:


# We want to see the holiday effects:
forecast_holidays[(forecast_holidays['playoff'] + forecast_holidays['superbowl']).abs() > 0][
        ['ds', 'yhat','playoff', 'superbowl']].tail()


# In[196]:


forecast_holidays_components= m_holidays.plot_components(forecast_holidays)


# Peyton won Superbowls: 2016. While losing 2010 and 2014.

# In[ ]:


df2_cv= cross_validation(m_holidays, horizon= '365 days', period= '180 days', initial= '730 days')


# In[203]:


metrics_2= performance_metrics(df2_cv)
display(metrics_2[['horizon', 'rmse', 'mape']])
print("Adding Holidays: Mean RMSE: {:.4f} Mean MAPE: {:.2%}".format(np.mean(metrics_2['rmse']), np.mean(metrics_2['mape'])))
print("Horizon of 365 days RMSE: {:.3f} and MAPE: {:.2%}".format(metrics_2.iloc[metrics_2.shape[0]-1]['rmse'], metrics_2.iloc[metrics_2.shape[0]-1]['mape']))


# In[256]:


h_1_mape= metrics_2.iloc[metrics_2.shape[0]-1]['mape']
h_1_rmse= metrics_2.iloc[metrics_2.shape[0]-1]['rmse']

plot_name= 'prophet.holidays.model.png'

forecast_holidays_plot= m_holidays.plot(forecast_holidays, figsize= (17,7))

ax = forecast_holidays_plot.gca()
ax.set_title("Model= Prophet(holidays), MAPE: {:.2%} and RMSE: {:.3f}".format(h_1_mape, h_1_rmse))
ax.set_xlabel("Time")
ax.set_ylabel("Wikipedia searches for Peyton (log)")

plt.show()
forecast_holidays_plot.savefig(image_dir+plot_name)


# # 3) Add another seasonality
# Besides yearly, weekly, and daily. Let's add monthly seasonality

# In[209]:


m_holidays_extra_s= Prophet(holidays= holidays)
# Increasing the number of Fourier components allows the seasonality to change more quickly (at risk of overfitting). Default values for yearly
# and weekly seasonalities are 10 and 3 respectively.
m_holidays_extra_s.add_seasonality(name= 'monthly', period= 30.5, fourier_order= 5)
m_holidays_extra_s.fit(peyton)


# In[210]:


fcst_month = m_holidays_extra_s.predict(future)


# In[213]:


monthly_components= m_holidays_extra_s.plot_components(fcst_month)


# In[ ]:


df3_cv= cross_validation(m_holidays_extra_s, horizon= '365 days', period= '180 days', initial= '730 days')


# In[217]:


metrics_3= performance_metrics(df3_cv)
display(metrics_3[['horizon', 'rmse', 'mape']])
print("Adding Holidays: Mean RMSE: {:.4f} Mean MAPE: {:.2%}".format(np.mean(metrics_3['rmse']), np.mean(metrics_3['mape'])))
print("Horizon of 365 days RMSE: {:.3f} and MAPE: {:.2%}".format(metrics_3.iloc[metrics_3.shape[0]-1]['rmse'], metrics_3.iloc[metrics_3.shape[0]-1]['mape']))


# In[225]:


h_1_mape= metrics_3.iloc[metrics_3.shape[0]-1]['mape']
h_1_rmse= metrics_3.iloc[metrics_3.shape[0]-1]['rmse']

# plot_name= 'initial.prophet.model.png'

extra_seasonality_plot= m_holidays_extra_s.plot(fcst_month, figsize= (17,7))

ax = extra_seasonality_plot.gca()
ax.set_title("Initial model= Prophet(holidays).add_seasonality, MAPE: {:.2%} and RMSE: {:.3f}".format(h_1_mape, h_1_rmse))
ax.set_xlabel("Time")
ax.set_ylabel("Wikipedia searches for Peyton (log)")

plt.show()


# # 4) Adding a regressor

# In[226]:


# creating an indicator variable for NFL sundays
def nfl_sundays(ds):
    """
    NFL season: August, September, October, November, December, January, February
    Sunday => 6 | range: 0-6 => Monday-Sunday
    """
    date= pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else: 
        return 0


# In[227]:


peyton['nfl_sunday']= peyton['ds'].apply(nfl_sundays)
peyton.head()


# In[228]:


m= Prophet()
m.add_regressor('nfl_sunday')
m.fit(peyton)


# In[231]:


# Regressor must also be available in future df
future['nfl_sunday'] = future['ds'].apply(nfl_sundays)
forecast = m.predict(future)
fig = m.plot_components(forecast)


# In[ ]:


df4_cv= cross_validation(m, horizon= '365 days', period= '180 days', initial= '730 days')


# In[244]:


metrics_4= performance_metrics(df4_cv)
display(metrics_4[['horizon', 'rmse', 'mape']])
print("Adding Holidays: Mean RMSE: {:.4f} Mean MAPE: {:.2%}".format(np.mean(metrics_4['rmse']), np.mean(metrics_4['mape'])))
print("Horizon of 365 days RMSE: {:.3f} and MAPE: {:.2%}".format(metrics_4.iloc[metrics_4.shape[0]-1]['rmse'], metrics_4.iloc[metrics_4.shape[0]-1]['mape']))


# In[245]:


h_1_mape= metrics_4.iloc[metrics_4.shape[0]-1]['mape']
h_1_rmse= metrics_4.iloc[metrics_4.shape[0]-1]['rmse']

# plot_name= 'initial.prophet.model.png'

regressor_plot= m.plot(forecast, figsize= (17,7))

ax = regressor_plot.gca()
ax.set_title("Initial model= Prophet() plus regressor, MAPE: {:.2%} and RMSE: {:.3f}".format(h_1_mape, h_1_rmse))
ax.set_xlabel("Time")
ax.set_ylabel("Wikipedia searches for Peyton (log)")

plt.show()


# ## We can specifiy if there's some change taking place
# For instance COVID

# In[25]:


# These are points where trend has changed
print('originally: ',m.changepoints[:5])

# you can specify changepoints if you want trend to only be allowed at certain points
m_c = Prophet(changepoints=['2014-01-01'])

print('\nnow: ',m_c.changepoints[:5])


# # 5) Combine Holidays and regressor

# In[249]:


m_holidays_regressor= Prophet(holidays= holidays)
fcst_holidays_regressor=  m_holidays_regressor.add_regressor('nfl_sunday').fit(peyton).predict(future)


# In[250]:


holidays_regressor= m_holidays_regressor.plot_components(fcst_holidays_regressor)


# In[ ]:


df5_cv= cross_validation(m_holidays_regressor, horizon= '365 days', period= '180 days', initial= '730 days')


# In[254]:


metrics_5= performance_metrics(df5_cv)
display(metrics_5[['horizon', 'rmse', 'mape']])
print("Adding Holidays: Mean RMSE: {:.4f} Mean MAPE: {:.2%}".format(np.mean(metrics_5['rmse']), np.mean(metrics_5['mape'])))
print("Horizon of 365 days RMSE: {:.3f} and MAPE: {:.2%}".format(metrics_5.iloc[metrics_5.shape[0]-1]['rmse'], metrics_5.iloc[metrics_5.shape[0]-1]['mape']))


# In[257]:


import session_info
session_info.show()


# In[ ]:




