#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[2]:


#preprocess
name = ["Age", "Education", "HusbandEducation", "Children", "Religion",
            "Working", "HusbandOccupation", "StandardLiving", "MediaExposure", "Contraceptive"]
dataset = pd.read_csv("cmc.data", names = name)
dataset


# In[3]:


dataArr = dataset.values
data = dataArr[:,0:9]  #data
label = dataArr[:,9]    # target


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)


# In[5]:


from sklearn.neighbors import KNeighborsClassifier


# In[6]:


# k = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# In[7]:


print(confusion_matrix(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[8]:


# k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# In[9]:


print(confusion_matrix(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[10]:


# k = 5
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# In[11]:


print(confusion_matrix(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

