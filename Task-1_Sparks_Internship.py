#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics
# # Task- 01
# # GRIPMAY21
# NAME -  Tanvi Bhute
# 
# TASK 1- Prediction using Supervised ML
# Predicting the percentage of a student from no. of study hours . Assuming the situation where a student studies for 9.25 hrs/day

# In[3]:


#Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[4]:


#importing dataset
dataset=pd.read_csv("http://bit.ly/w-data")
print("Data has been imported successfully!")
dataset.head()


# In[5]:


dataset.dtypes


# In[6]:


dataset.shape


# In[7]:


dataset.describe()


# In[8]:


#plotting the distribution of scores
dataset.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Percentage')
plt.xlabel('Study Hours')
plt.ylabel('Percentage scored')
plt.show()


# # The above graph repreesnts a positive linear regression between Study Hours and percentage scored
# # Preparing data - dividing data into attributes and labels

# In[9]:


#selecting dependent and independent variables from dataset
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values
print(y)


# In[10]:


#splitting in training and testing dataset
X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size = 0.2, random_state = 0)


# In[11]:


#training linear Regreesion Algorithm
reg = LinearRegression()
reg.fit(X_train , Y_train)


# In[12]:


#Plotting Regression line
plt.scatter(X_train,Y_train, color = 'green')
plt.title('Training set')
plt.plot(X_train, reg.predict(X_train))
plt.xlabel('Study Hours')
plt.ylabel('Percentage scored')
plt.show()


# In[13]:


#accuracy of training process
reg.score(X_train,Y_train)


# In[14]:


#Prediction of testing process
Y_pred = reg.predict(X_test)


# In[15]:


#visualizing the testing rocess results
plt.scatter(X_test , Y_test , color = 'red')
plt.plot(X_train , reg.predict(X_train),color = 'blue')
plt.title('Testing set')
plt.xlabel('Study Hours')
plt.ylabel('Percentage scored')
plt.show()


# In[16]:


print(X_test)#Testing data
y_pred = reg.predict(X_test)#Prediction of scores


# # Main Task - predicting percentage of student studying 9.25hrs/day 

# In[17]:


hours = 9.25
pred = reg.predict([[hours]])
print("No. of Hours = {}".format(hours))
print("Score predicted = {}".format(pred[0]))


# # Therefore the percentage score will be 93.7%

# # Evaluating the Model
# 
# # This is the final step to evaluate our model.This represents how well different algorithm perform on particular dataset.For example , here we have chosen mean square error

# In[19]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test, Y_pred))


# In[20]:


#Visualizing Trainig set
plt.scatter(X_train,Y_train)
plt.title('Training set')
plt.plot(X_train,reg.predict(X_train))
plt.xlabel('Study Hours')
plt.ylabel('Percentage Scored')
plt.show()


# #  Task 1 completed!
