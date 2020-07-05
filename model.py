#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:


data = pd.read_csv(r'C:\Users\AVINESH\Downloads\datasets_192683_428563_hiring.csv')


# In[3]:


data


# In[4]:


data['experience'].fillna(0, inplace=True)

data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean(), inplace=True)


# In[5]:


data


# In[6]:


X =data.iloc[:,:3]


# In[7]:


def convert_to_int(word):
    
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0, 0: 0}

    return word_dict[word]    


# In[8]:


X['experience']= X['experience'].apply(lambda x : convert_to_int(x))

y = data.iloc[:,-1]


# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[10]:


regressor.fit(X,y)


# In[11]:


pickle.dump(regressor,open('model.pkl','wb'))


# In[12]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))


# In[ ]:




