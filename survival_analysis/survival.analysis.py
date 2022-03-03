#!/usr/bin/env python
# coding: utf-8

# # Survival Analysis
# March 3th 2022

# In[21]:


import sys
print(sys.executable)
import numpy as np
import pandas as pd
import os
print(os.getcwd())
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# Survival modules:
from lifelines import KaplanMeierFitter, CoxPHFitter


# In[2]:


# Custom functions:
import src.colorsetup


# ## Input Data

# In[6]:


data_path= 'course_data/'
df= pd.read_pickle(data_path+'churndata.pkl')
display(df.head())
print(df.shape)


# In[11]:


# Plot churn values
print("Percentage of churn: {:.4f}".format(df.churn_value.sum()/df.shape[0]))
sns.barplot(data= df, x= 'churn_value', y= 'months')
plt.xlabel("Churn values")
plt.ylabel("Time (months)")
plt.show()


# We observe as expected that months are shorter for churns

# ## Plot the Kaplan-Meier Curve

# In[20]:


kmf= KaplanMeierFitter()
kmf.fit(df.months, df.churn_value, label= "Kaplan-Meier Curve")
kmf.plot(linewidth= 4)
plt.title("Customer churn")
plt.xlabel("Time (months)")
plt.ylabel("Survival probability")
plt.show()


# In[53]:


# Let's inspect multiple services variable
print(Counter(df.multiple))
df1= df[df.multiple == 'Yes']
df2= df[df.multiple == 'No']

fig, axes= plt.subplots(1,2)
fig.set_figwidth(18)

df1.groupby('churn_value')['months'].plot(kind= 'hist', ax= axes[0], title= "Customers with Multiple Services")
axes[0].legend(labels= ['Not churned', 'Churned'])
df2.groupby('churn_value')['months'].plot(kind= 'hist', ax= axes[1], title= "Customers with Single Service")

plt.show()


# In[51]:


kmf.fit(df1.months, df1.churn_value)
kmf.plot(label= "Multiple Services")
kmf.fit(df2.months, df2.churn_value)
kmf.plot(label= "Single Service")
plt.title("Assess Services")
plt.ylabel('Survival probability')
plt.xlabel('Time (months)')
plt.show()


# ## Cox Proportional Hazard model

# In[65]:


dfu= df[['multiple', 'churn_value']]
display(dfu.head())
dfd= pd.get_dummies(dfu, drop_first= True)
display(dfd.head())
dfd['months']= df.months
dfd.rename(columns= {'multiple_Yes': 'multiple_services'}, inplace= True)
dfd.head()


# In[66]:


cph= CoxPHFitter()
cph.fit(dfd, duration_col= 'months', event_col= 'churn_value')
cph.print_summary()


# P-value is significant, and multitple_services coefficient is negative which means it's important (the lower => the better)

# In[68]:


cph.plot()
plt.show()


# In[79]:


cph.plot_partial_effects_on_outcome(covariates= 'multiple_services', values= [1,0], plot_baseline= False, lw= 4)
plt.show()


# ## Use more variables to fit Cox-proportional Hazard model

# In[89]:


df_more_var= df[['churn_value', 'satisfaction', 'security', 'backup', 'support']]
display(df_more_var.head())
dummy_df= pd.get_dummies(df_more_var, drop_first= True)
display(dummy_df.head())
rename_columns= {'security_Yes': 'security_service', 'backup_Yes': 'backup_service', 'support_Yes': 'support_service'}
dummy_df.rename(columns= rename_columns, inplace= True)
dummy_df['months']= df.months
dummy_df.head()


# In[91]:


new_model= CoxPHFitter().fit(dummy_df, duration_col= 'months', event_col= 'churn_value')
new_model.print_summary()


# In[96]:


print("Satisfaciton is the most important feature for our model")
new_model.plot()
plt.show()


# In[97]:


print("There's a clear difference among the levels of satisfaction")
new_model.plot_partial_effects_on_outcome(covariates= 'satisfaction', values= [5,4,3,2,1], plot_baseline= False, lw= 4)
plt.show()


# In[ ]:





# In[ ]:




