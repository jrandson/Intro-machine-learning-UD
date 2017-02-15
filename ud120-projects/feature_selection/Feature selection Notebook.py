
# coding: utf-8

# In[1]:

import sklearn.linear_model.Lasso

features, labels = GetMyData()

regression  = Lasso()
regression.fit(features)


