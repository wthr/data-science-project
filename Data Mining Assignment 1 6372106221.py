#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df.head()


# In[7]:


df['Attrition'].replace('Yes',1,inplace=True)
df['Attrition'].replace('No',0,inplace=True)
df['OverTime'].replace('Yes',1,inplace=True)
df['OverTime'].replace('No',0,inplace=True)
df.head()


# In[9]:


df.describe(include='all')


# In[10]:


df.nunique()


# In[11]:


df1 = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
df1.head()


# In[13]:


df2 = pd.get_dummies(df1)
df2.head()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support


# In[15]:


X = df2.drop('Attrition', axis=1)
X


# In[16]:


y = df2['Attrition']
y


# In[17]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size=0.3,random_state=42)


# In[18]:


tree = DecisionTreeClassifier()
tree.fit(X_trainset,y_trainset)


# In[19]:


pred= tree.predict(X_testset)
pred[0:5]


# In[21]:


trainpred = tree.predict(X_trainset)
trainpred[0:5]


# In[22]:


y_trainset.value_counts()


# In[23]:


pd.Series(trainpred).value_counts()


# In[24]:


y_testset.value_counts()


# In[25]:


pd.Series(pred).value_counts()


# In[28]:


metrics.accuracy_score(y_trainset,trainpred)


# In[29]:


metrics.accuracy_score(y_testset, pred)


# In[30]:


precision_recall_fscore_support(y_trainset,trainpred)


# In[31]:


precision_recall_fscore_support(y_testset,pred)


# In[ ]:




