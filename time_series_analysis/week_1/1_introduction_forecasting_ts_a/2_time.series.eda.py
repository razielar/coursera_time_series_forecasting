#!/usr/bin/env python
# coding: utf-8

# # Time-series EDA
# Exploratory Data Analysis (EDA)  
# Feb 16th 2021

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Custom functions:
import src.colorsetup
data_path= '../course_data/1_intro_forecasting_ts_analysis/'


# ## Input Data

# In[4]:


df= pd.read_excel(data_path+'sample-superstore.xls')
print(df.shape)
df.head()


# ## Pre-processing
# Set-up data monthly

# In[5]:


# Select data
new_vars= ['Segment', 'Profit', 'Order Date', 'Sales']
new_base= df[new_vars].set_index('Order Date')
display(new_base.head())
# Remove duplicated rows
prof_pivot= new_base.pivot_table(columns= "Segment", index= "Order Date")
display(prof_pivot.head())
# Set-up data monthly: sum as an aggregation function
prof_month= prof_pivot.resample('M').sum()
prof_month.head(12)


# In[6]:


prof_month.index


# ## Visualizations

# In[15]:


fig, axes= plt.subplots(3,2)
fig.set_figwidth(16)
fig.set_figheight(7)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        axes[i,j].plot(prof_month[money, cat])
        axes[i,j].title.set_text(cat+" "+money)
        axes[i,j].set_ylabel(money)
    
fig.tight_layout()
plt.show()


# In[17]:


fig, axes= plt.subplots(3,2)
fig.set_figwidth(16)
fig.set_figheight(7)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        plot_pacf(prof_month[money, cat], ax= axes[i,j], title= "PACF-"+money+": "+cat, lags= 12)
    
fig.tight_layout()
plt.show()


# In[18]:


fig, axes= plt.subplots(3,2)
fig.set_figwidth(16)
fig.set_figheight(7)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        month_plot(prof_month[money, cat], ax= axes[i,j], ylabel= money)
    
fig.tight_layout()
plt.show()


# Develop a function for EDA

# In[27]:


cat_var = 'Region'
date_var = 'Order Date'
money_vars = ['Profit', 'Sales']

def monthly_eda(cat_var=cat_var,
                date_var=date_var, 
                money_vars=money_vars):
    new_vars = [cat_var, date_var] + money_vars
    cats = list(df[cat_var].unique())
    num_cats = len(cats)
    new_base = df[new_vars].set_index(date_var)
    prof_pivot = new_base.pivot_table(columns=cat_var,index = date_var)
    prof_month = prof_pivot.resample('M').sum()
    prof_month.head()

    fig,axes = plt.subplots(num_cats*3, 2, figsize=(16, 5*num_cats),)
    for i,cat in enumerate(cats):
        for j,money in enumerate(money_vars):
            axes[i,j].plot(prof_month[money,cat])
            axes[i,j].title.set_text(cat+" "+money)
            fig = plot_pacf(prof_month[money,cat],ax=axes[i+num_cats,j],title = cat+" "+money+" PACF")
            fig = month_plot(prof_month[money,cat],ax=axes[i+num_cats*2,j])
            axes[i+num_cats*2,j].title.set_text(cat+" Seasonality")

    fig.tight_layout()
    plt.show()


# In[28]:


monthly_eda(cat_var='Region')


# In[29]:


import session_info
session_info.show()


# In[ ]:




