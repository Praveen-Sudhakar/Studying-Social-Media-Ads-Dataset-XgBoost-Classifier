#!/usr/bin/env python
# coding: utf-8

# In[148]:


#Import necessary packages

import numpy as np
import pandas as pd


# In[149]:


#Reading the data from the stored variable

df = pd.read_csv("D:\AIML\Dataset\Social_Network_Ads.csv")

df


# In[150]:


df.info()


# In[151]:


#Declaring IV

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1]


# In[152]:


print(x[0:5])
print(y[0:5])


# In[153]:


print(set(df['Gender']))
print(set(df['Purchased']))


# In[156]:


#Label Encoding the 'Objects' columns 

from sklearn.preprocessing import LabelEncoder

gend = LabelEncoder()

gend.fit(['Male', 'Female'])

x[:,0] = gend.transform(x[:,0])


# In[157]:


x[0:5]


# In[158]:


#Splitting the dataset

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=100)

print("Train shape",train_x.shape,train_y.shape)
print("Test shape",test_x.shape,test_y.shape)


# In[159]:


#Modeling

from xgboost import XGBClassifier

xgcl = XGBClassifier()

xgcl.fit(train_x,train_y)


# In[161]:


#Evaluating using test dataset

pred_y = xgcl.predict(test_x)


# In[168]:


#Checking accuracy score

from sklearn import metrics

print(f"Accuracy score = {metrics.accuracy_score(pred_y,test_y)*100} %")


# In[176]:


#Plotting the graph

import matplotlib.pyplot as plt

plt.scatter(test_x[:,1],test_x[:,2],c=test_y)


# In[179]:


plt.scatter(test_x[:,1],test_x[:,2],c=pred_y)


# In[ ]:




