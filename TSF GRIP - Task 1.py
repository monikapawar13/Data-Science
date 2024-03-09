#!/usr/bin/env python
# coding: utf-8

# # GRIP_March2024 : The Sparks Foundation

# # Task 1: Prediction using Supervised ML

# # Author: Monika Pawar

# # Percentage scored by a student based on study hours

# In[3]:


#importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Reading the Dataset

url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.corr()


# In[12]:


#Plotting the distribution graph of scores
plt.style.use('ggplot')
df.plot(kind='line')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[15]:


xmin=min(df.Hours)
xmax=max(df.Hours)
df.plot(kind='area',alpha=0.8,stacked=True,figsize=(10,5),xlim=(xmin,xmax))
plt.title('Hors vs Score',size=15)
plt.xlabel('Hours',size=15)
plt.ylabel('Score',size=15)
plt.show()


# In[14]:


df.plot(kind='scatter',x='Hours',y='Scores',color='r',figsize=(8,5))
plt.title('Hours vs Percentage')
plt.xlabel('Hors Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[17]:


x=np.asanyarray(df[['Hours']])
y=np.asanyarray(df['Scores'])

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=2)

regressor = LinearRegression()
regressor.fit(train_x,train_y)

print('Traning Completed\n')
print('Coefficients:',regressor.coef_)
print('Intercept:',regressor.intercept_)


# In[19]:


df.plot(kind='scatter',x='Hours',y='Scores',figsize=(10,5),color='r')
plt.plot(train_x,regressor.coef_[0]*train_x + regressor.intercept_,color='b')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[20]:


from sklearn import metrics
from sklearn.metrics import r2_score

y_pred=regressor.predict(test_x)
print('Mean Absolute Error : {}',format(metrics.mean_absolute_error(y_pred,test_y)))
print("R2-score: %.2f" % r2_score(y_pred,test_y))


# In[22]:


df2 = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})
df2


# In[23]:


hours=9.25
predicted_score=regressor.predict([[hours]])

print(f'No. of hours = {hours}')
print(f'Predicted Score = {predicted_score[0]}')


# In[24]:


df.plot(kind='bar')


# In[ ]:




