
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

my_data = pd.read_csv('data2.txt',names=["tamaño","habitaciones","precio"]) 


# In[5]:


print(my_data)


# In[22]:


my_data.std()


# In[23]:


my_data.mean()


# In[25]:


my_data = (my_data - my_data.mean())/my_data.std()
print(my_data)


# In[48]:


X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = my_data.iloc[:,2:3].values 
theta = np.zeros([1,3])

alpha = 0.03
iters = 500


# 
# 

# In[40]:




def funcionCosto(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
funcionCosto(X,y,theta)


# In[49]:




def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = funcionCosto(X, y, theta)
        
    
    return theta,cost


g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = funcionCosto(X,y,g)
print(finalCost)

