#!/usr/bin/env python
# coding: utf-8

# # Time-series EDA
# Exploratory Data Analysis (EDA)  
# Feb 16th 2021

# In[17]:


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
# Config:
pd.options.display.float_format = '{:,.2f}'.format
plotsize = (20, 8)
sns.set_context("paper", font_scale= 1.3)
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.spines.top']= False
plt.rcParams['figure.figsize']= plotsize


# ## Input Data

# In[2]:


df= pd.read_excel('../course_data/sample-superstore.xls')
print(df.shape)
df.head()


# ## Pre-processing
# Set-up data monthly

# In[9]:


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


# In[12]:


prof_month.index


# ## Visualizations

# In[16]:


fig, axes= plt.subplots(3,2)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        axes[i,j].plot(prof_month[money, cat])
        axes[i,j].title.set_text(cat+" "+money)
        axes[i,j].set_ylabel(money)
    
fig.tight_layout()
plt.show()


# In[18]:


fig, axes= plt.subplots(3,2)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        plot_pacf(prof_month[money, cat], ax= axes[i,j], title= "PACF-"+money+": "+cat, lags= 12)
    
fig.tight_layout()
plt.show()


# In[19]:


fig, axes= plt.subplots(3,2)

for i, cat in enumerate(['Consumer', 'Corporate', 'Home Office']):
    for j, money in enumerate(['Sales', 'Profit']):
        month_plot(prof_month[money, cat], ax= axes[i,j], ylabel= money)
    
fig.tight_layout()
plt.show()


# Develop a function for EDA

# In[22]:


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

    fig,axes = plt.subplots(num_cats*3, 2, figsize=(20, 5*num_cats),)
    for i,cat in enumerate(cats):
        for j,money in enumerate(money_vars):
            axes[i,j].plot(prof_month[money,cat])
            axes[i,j].title.set_text(cat+" "+money)
            fig = plot_pacf(prof_month[money,cat],ax=axes[i+num_cats,j],title = cat+" "+money+" PACF")
            fig = month_plot(prof_month[money,cat],ax=axes[i+num_cats*2,j])
            axes[i+num_cats*2,j].title.set_text(cat+" Seasonality")

    fig.tight_layout()
    plt.show()


# In[23]:


monthly_eda(cat_var='Region')


# In[ ]:





# In[ ]:




