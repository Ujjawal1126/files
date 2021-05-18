#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation - Internship (Task - 1)

# # Name - Ujjawal Kumar Nayak   (grip May'21)

#      

# # Prediction using Supervised ML

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Reading data  

# In[2]:


url = 'http://bit.ly/w-data'
df = pd.read_csv(url)
df.head()


# Plotting the raw data to check the nature of dependency i.e. linear etc

# In[3]:


plt.figure(figsize=(10,5))
plt.scatter(df.Hours,df.Scores)
plt.xlabel('Hours of study')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.xlim((0,10))
plt.ylim((0,100))
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,100,5))
plt.grid()
plt.show()


# As we see, in above plot that there is linear relationships between Score and hours of study

# So, we have to train the data according to above. So, our further step to train the data

# # Training the dataset

# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Hours.values.reshape(-1,1),df.Scores.values.reshape(-1,1), test_size = 0.2)


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


model = LinearRegression()


# In[7]:


model.fit(X_train,y_train)


# In[8]:


pred = model.predict(X_test)


# # Prediction

# In[9]:


pred


# In[10]:


y_test


# In[11]:


model.score(X_test,y_test)


# As we are getting 94.8% efficiency of the model then it show our program is good. It will predict nicely. Now let us check in plot how it is looking

# In[12]:


plt.figure(figsize=(10,5))
plt.plot(X_train,model.coef_*X_train+model.intercept_,color = 'black',label = 'Prediction Line')
plt.scatter(X_train,y_train,color='grey',label = 'Training Data')
plt.scatter(X_test,y_test,color='red',label = 'Testing Data')
plt.scatter(X_test,pred,color='green',label = 'Predictions')
plt.xlabel('Hours of study')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.xlim((0,10))
plt.ylim((0,100))
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,100,5))
plt.legend()
plt.grid()
plt.show()


# # Prediction with given input

# In[14]:


model.predict([[9.25]])


# Here we get Score 92.78 after study of 9.25 hours of study as per question.
