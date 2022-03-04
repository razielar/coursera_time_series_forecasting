#!/usr/bin/env python
# coding: utf-8

# # Playground to understand Differencing
# Feb 24th 2022

# In[3]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# Import custom functions
import src.colorsetup
pd.options.display.float_format = '{:,.4f}'.format


# In[5]:


play= pd.DataFrame([[i for i in range(1,11)], [i**2 for i in range(1,11)]]).T
play.columns = ["original", "squared"]
play


# In[6]:


play.original.diff()


# In[9]:


fig, axes = plt.subplots(1,3, figsize= (18,5))

axes[0].plot(play.squared)
axes[0].set_title("Squared")
axes[1].plot(play.squared.diff())
axes[1].set_title("First diff")
axes[2].plot(play.squared.diff().diff())
axes[2].set_title("Second diff")

plt.show()


# In[10]:


play.squared.diff().diff()


# In[11]:


import session_info
session_info.show()


# In[ ]:




