#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np


# In[2]:
get_ipython().system('dir')


# In[3]:
name = ["Age", "Education", "HusbandEducation", "Children", "Religion",
            "Working", "HusbandOccupation", "StandardLiving", "MediaExposure", "Contraceptive"]


# In[4]:
name


# In[5]:
dataset = pd.read_csv("cmc.data", names = name)


# In[6]:
dataset


# In[7]:
dataArr = dataset.values
dataArr


# In[8]:
data = dataArr[:,0:9]  #data
label = dataArr[:,9]    # target


# In[9]:
data


# In[10]:
label


# In[33]:
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
model = ComplementNB()
model2 = GaussianNB()
model3 = MultinomialNB()
model4 = BernoulliNB()


# In[26]:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[27]:
cross_val_predict(model, data, label, cv=10)


# In[34]:
accuracy = cross_val_score(model, data, label, cv=10)
accuracy2 = cross_val_score(model2, data, label, cv=10)
accuracy3 = cross_val_score(model3, data, label, cv=10)
accuracy4 = cross_val_score(model4, data, label, cv=10)


# In[23]:
accuracy


# In[24]:
max(accuracy)


# In[35]:
max(accuracy2)


# In[36]:
max(accuracy3)


# In[37]:
max(accuracy4)