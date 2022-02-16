#!/usr/bin/env python
# coding: utf-8

# # Pandas time-series
# Feb 15th 2022

# In[85]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# Config:
pd.options.display.float_format = '{:,.2f}'.format
plotsize = (13, 5)


# ## Read data
# 4 years of daily sales

# In[7]:


df= pd.read_excel('data/sample-superstore.xls')
print(df.shape)
df.head()


# In[7]:


# Pring column names:
for i,j in enumerate(df.columns):
    print(i+1, j)


# ### Simplify Time-series data
# Total Sales by order date and category

# In[17]:


variables= ['Order Date', 'Category', 'Sales']
df.loc[:, variables]


# In[36]:


# Group_by date and category and important feat sales:
base= df.groupby(['Order Date', 'Category'], as_index= False)['Sales'].sum()
base


# In[37]:


base.dtypes


# Let's check the number of years we have in this dataset

# In[47]:


# Unique years we have in the dataset:
np.unique(np.array(base['Order Date'], dtype= 'datetime64[Y]')) 


# In[51]:


# Monthly
tmp= np.unique(np.array(base['Order Date'], dtype= 'datetime64[M]'))
tmp


# In[56]:


print('Number of months:', len(tmp))


# In[77]:


print(base['Category'].unique())


# ## Working with the Pandas DatetimeIndex

# In[67]:


# Way easier to work with DatetimeIndex
base.set_index('Order Date', inplace= True)


# In[73]:


base


# In[91]:


display(base.loc['2011'].tail())
display(base[base['Category'] == 'Technology'].loc['2011'])
display(base[base['Category'] == 'Technology'].loc['2011':'2012-04'])


# In[97]:


# Day of week= Monday=0, Sunday= 6
base['DayofWeek']= base.index.dayofweek 
display(base)
del(base['DayofWeek'])


# ## Standardizing the DatetimeIndex
# Some time-series applications require that data contain all periods and have a frequency assigned (freq= None)   
# We need to ensure there are:
# * No duplicate index values: **Pivot data**
# * No missing index values

# In[100]:


display(base.index)


# In[104]:


base.reset_index(inplace= True)
sales= base.pivot(index= 'Order Date', columns= 'Category', values= 'Sales').fillna(0)
sales


# In[111]:


print(sales.index)
print('Unique values:', len(sales.index.unique()))


# ## Generating a complete Index and Setting Frequency
# Since we're using daily data, we would like to set a daily frequency

# In[119]:


print(len(sales.index.unique()))
print(sales.index.min())
print(sales.index.max())
date_range= sales.index.max() - sales.index.min()
date_range


# In[121]:


new_index= pd.date_range(sales.index.min(), sales.index.max())
new_index


# In[122]:


new_sales= sales.reindex(new_index, fill_value= 0)
new_sales


# We can observe we have a daily frequency

# In[124]:


new_sales.index


# In[128]:


new_sales[new_sales['Furniture'] == 0]


# In[131]:


# new_sales.to_csv('processed_df/processed.df.csv')


# In[ ]:





# In[ ]:





# In[ ]:




