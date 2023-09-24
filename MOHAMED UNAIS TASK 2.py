#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


card_df=pd.read_csv('creditcard.csv')


# In[4]:


card_df


# In[5]:


card_df.describe()


# In[6]:


card_df.info()


# In[7]:


card_df.shape


# In[8]:


card_df.isnull().sum()


# In[9]:


card_df['Class'].value_counts()


# In[10]:


legit= card_df[card_df.Class==0]
fraud= card_df[card_df.Class==1]
print(legit.shape)
print(fraud.shape)


# In[11]:


sns.set(style="whitegrid")

ax = sns.countplot(x='Class', data=card_df, order=[0, 1], label='Count', palette='pastel')

counts = card_df['Class'].value_counts()


for i, count in enumerate(counts):
    ax.text(i, count, str(count), ha='center', va='bottom', fontsize=12)

title = "The Amount of Fraud and Non Fraud Transactions"
ax.set_title(title, size=16)
plt.title(title, size=16, pad=20)


plt.show()


# In[12]:


legit_sample=legit.sample(n=492)


# In[13]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[14]:


new_dataset.head(5)


# In[15]:


new_dataset.tail(5)


# In[16]:


new_dataset['Class'].value_counts()


# In[17]:


X=new_dataset.drop(columns='Class',axis=0)
Y=new_dataset['Class']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=2)
print(X.shape,X_train.shape,X_test.shape)


# In[23]:


model = LogisticRegression(max_iter=1000)


# In[24]:


model.fit(X_train,Y_train)


# In[25]:


LogisticRegression()


# In[26]:


X_train_predction=model.predict(X_train)
training_accuracy=accuracy_score(X_train_predction,Y_train)


# In[27]:


print('Accoracy of training model :',training_accuracy)


# In[28]:


X_test_predction=model.predict(X_test)
testing_accuracy=accuracy_score(X_test_predction,Y_test)
print('Accoracy of testing model :',testing_accuracy)

