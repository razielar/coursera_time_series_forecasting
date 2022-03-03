#!/usr/bin/env python
# coding: utf-8

# # Construct Trend, Seasonality, and Residual components
# Feb 17th 2021

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
SEED=10


# In[3]:


# Custom functions:
import src.colorsetup
data_path= '../course_data/1_intro_forecasting_ts_analysis/'


# In[4]:


time= np.arange(1,51)
time


# In[5]:


trend= time * 2.75
trend


# In[6]:


plt.plot(time, trend, 'b.')
plt.title('Trend vs Time')
plt.xlabel('Time (min)')
plt.ylabel('Electricity')
plt.show()


# In[7]:


seasonal= 10+np.sin(time)*10
plt.plot(time, seasonal, 'g-.')
plt.title('Seasonality vs. Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Electricity')
plt.show()


# In[8]:


np.random.seed(SEED)
residual= np.random.normal(loc= 0.0, scale= 1, size= len(time)) #mean= 0, and sd= 1
plt.plot(time, residual)
plt.title('Residuals vs. Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Electricity')
plt.show()


# ## Aggregate trend, seasonality, and residuals

# ### Additive model

# In[9]:


additive= trend + seasonal + residual
plt.plot(time, additive)
plt.title('Additive model')
plt.xlabel('Time (minutes)')
plt.ylabel('Electricity')
plt.show()


# ### Multiplicative models

# In[10]:


ignore_residuals= np.ones_like(residual)
multiplicative= trend * seasonal * ignore_residuals
plt.plot(time, multiplicative)
plt.title('Multiplicative model')
plt.xlabel('Time (minutes)')
plt.ylabel('Electricity')
plt.show()


# ## Analysis of new data

# In[11]:


time= np.arange(0, 50)

data_path= '../course_data/2_time_series-decomposition-material/'
data_a= np.load(data_path+"dataset_A.npy")
data_b= np.load(data_path+"dataset_B.npy")


# In[12]:


plt.plot(time, data_a)
plt.title("Probably: multiplicative")
plt.grid(alpha= 0.4)
plt.show()


# In[13]:


plt.plot(time, data_b)
plt.title("Probably: additive")
plt.grid(alpha= 0.4)
plt.show()


# In[14]:


import session_info
session_info.show()


# In[ ]:





# In[ ]:




