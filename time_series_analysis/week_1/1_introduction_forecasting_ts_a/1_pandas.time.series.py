#!/usr/bin/env python
# coding: utf-8

# # Pandas time-series
# Feb 15th 2022

# In[5]:


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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot


# In[8]:


# Custom functions:
import src.colorsetup
data_path= '../course_data/1_intro_forecasting_ts_analysis/'


# ## Read data
# 4 years of daily sales

# In[9]:


df= pd.read_excel(data_path+'sample-superstore.xls')
print(df.shape)
df.head()


# In[10]:


# Pring column names:
for i,j in enumerate(df.columns):
    print(i+1, j)


# ### Simplify Time-series data
# Total Sales by order date and category

# In[11]:


variables= ['Order Date', 'Category', 'Sales']
df.loc[:, variables]


# In[12]:


# Group_by date and category and important feat sales:
base= df.groupby(['Order Date', 'Category'], as_index= False)['Sales'].sum()
base


# In[13]:


base.dtypes


# Let's check the number of years we have in this dataset

# In[14]:


# Unique years we have in the dataset:
np.unique(np.array(base['Order Date'], dtype= 'datetime64[Y]'))


# In[15]:


# Monthly
tmp= np.unique(np.array(base['Order Date'], dtype= 'datetime64[M]'))
tmp


# In[16]:


print('Theoretical num. months:', 12*4)
print('Number of months:', len(tmp))


# In[17]:


print(base['Category'].unique())


# ## Working with the Pandas DatetimeIndex

# In[18]:


# Way easier to work with DatetimeIndex
base.set_index('Order Date', inplace= True)


# In[19]:


base


# In[20]:


display(base.loc['2011'].tail())
display(base[base['Category'] == 'Technology'].loc['2011'])
display(base[base['Category'] == 'Technology'].loc['2011':'2012-04'])
display(base[(base['Category'] == 'Technology') | (base['Category'] == 'Furniture')].loc['2014'])


# In[21]:


# Day of week= Monday=0, Sunday= 6
base['DayofWeek']= base.index.dayofweek 
display(base)
del(base['DayofWeek'])


# ## Standardizing the DatetimeIndex
# Some time-series applications require that data contain all periods and have a frequency assigned (freq= None)   
# We need to ensure there are:
# * No duplicate index values: **Pivot data**
# * No missing index values

# In[22]:


display(base.index)


# In[23]:


base.reset_index(inplace= True)
sales= base.pivot(index= 'Order Date', columns= 'Category', values= 'Sales').fillna(0)
sales


# In[24]:


print(sales.index)
print('Unique values:', len(sales.index.unique()))


# ## Generating a complete Index and Setting Frequency
# Since we're using daily data, we would like to set a daily frequency

# In[25]:


print(len(sales.index.unique()))
print(sales.index.min())
print(sales.index.max())
date_range= sales.index.max() - sales.index.min()
date_range


# In[26]:


new_index= pd.date_range(sales.index.min(), sales.index.max())
new_index


# In[27]:


new_sales= sales.reindex(new_index, fill_value= 0)
new_sales


# We can observe we have a daily frequency

# In[28]:


new_sales.index


# In[29]:


new_sales[(new_sales['Furniture'] == 0) & (new_sales['Technology'] == 0) & (new_sales['Office Supplies'] == 0)]


# In[131]:


# new_sales.to_csv('processed_df/processed.df.csv')


# ## Resampling
# Upsampling= moving to a lower frequency (i.e. from days to weeks)  
# Downsampling= moving to a higher frequency (i.e. from years to months)

# ### Upsampling

# In[30]:


display(new_sales)
sales_weekly= new_sales.resample('W').sum()
display(sales_weekly)


# In[31]:


print('Original DF')
display(new_sales)
# Monthly:
print('Monthly Sales')
sales_monthly= new_sales.resample('M').sum()
display(sales_monthly.head())
print(sales_monthly.shape)
# Quartely:
sales_quarterly = new_sales.resample('Q').sum()
print('Quarterly Sales')
display(sales_quarterly.head())
print(sales_quarterly.shape)
# Annual:
sales_annual = new_sales.resample('Y').sum()
print('Annual Sales')
display(sales_annual.head())
print(sales_annual.shape)


# ### Downsampling
# Moving from annual to monthly for example, requires an option to fill in missing values. A common approach is the interpolate method (i.e. linear, spline, etc.)

# In[32]:


display(sales_annual.resample('M').sum())
sales_monthly_from_annual= sales_annual.resample('M')
sales_monthly_from_annual.interpolate(method= 'spline', order= 3)


# ## Variable Transformation
# Such as: log, differences, growth rate, etc.

# In[33]:


### Difference
display(sales_monthly.head(12))
sales_monthly.diff().head(12)


# In[34]:


## Percentage difference
display(sales_monthly.head(12))
display(sales_monthly.pct_change().head(12))
sales_monthly.join(sales_monthly.pct_change().add_suffix('_%_change')).head(12)


# In[35]:


display(sales_monthly.head(12))
np.log(sales_monthly+1).head(12) 


# ### Rolling Average and Windows
# Smoothing our data

# In[36]:


print(new_sales.index.min())
print(np.mean([2573.82,76.73,0,0,0,0,0]).round(2))
window_size= 7
rolling_window= new_sales.rolling(window= window_size)
# Average each week
display(new_sales.head(7))
rolling_window.mean().dropna().head()


# In[37]:


rolling_window.std().dropna().head(window_size)


# In[38]:


display(new_sales.head())
new_sales.cumsum().dropna().head()


# ## Visualization

# In[39]:


# new_sales.plot()
sales_weekly.plot(title= 'Sales Weekly')

sales_monthly.plot(title= 'Sales Monthly')

sales_quarterly.plot(title= 'Sales Quartely')

sales_annual.plot(title= 'Sales Annual')

plt.show()


# In[40]:


rolling_window.std().plot(title='Daily Sales Standard Deviation, 7-day Rolling Average')

# Monthly Sales Percent Change
sales_monthly.pct_change().plot(title='Monthly Sales % Change')

# Cumulative Weekly Sales
sales_weekly.cumsum().plot(title='Cumulative Weekly Sales')

# Quarterly Sales Growth
sales_quarterly.pct_change().plot(title='Quarterly Sales % Change')

plt.show()


# ## Time-series visualizations
# ACF and PACF= if there's a correlation between one period and the next  
# ACF= we care about the indirect and direct effects  
# PACF= we only care about the direct effect => **AR models**

# In[41]:


# How much it correlate with the past value
display(new_sales.head())
pacf_plot= plot_pacf(new_sales['Furniture'], lags= 30, title= "Partial Autocorrelation in Furniture Daily Sales Data")
acf_plot= plot_acf(new_sales['Furniture'], lags= 30, title= "Autocorrelation in Furniture Daily Sales Data")


# In[42]:


display(sales_weekly.head())
pacf_plot = plot_pacf(sales_weekly['Furniture'], lags=12, title='Partial Autocorrelation in Furniture Weekly Sales Data')
acf_plot = plot_acf(sales_weekly['Furniture'], lags=12, title='Autocorrelation in Furniture Weekly Sales Data')


# In[52]:


image_path= '../../../plots/time_series_analysis/'
plot_name= 'seasonal.plot.png'

display(sales_monthly.head(12))
m_plot = month_plot(sales_monthly['Furniture'])
m_plot.savefig(image_path+plot_name, transparent= True, bbox_inches= 'tight')


# In[44]:


display(sales_quarterly.head(8))
q_plot= quarter_plot(sales_quarterly['Furniture'])


# In[45]:


import session_info
session_info.show()


# In[ ]:




