
# coding: utf-8

# In[5]:

from  sklearn  import linear_model
clf = linear_model.LinearRegression()
clf.fit([[1,3],[2,4],[3,7]],[5,8,13])
print clf.coef_



# ## Quiz Regressio Model 

# In[10]:

def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    from sklearn import linear_model
    ### name your regression reg
    reg = linear_model.LinearRegression()
    ### your code goes here!
    reg.fit(ages_train, net_worths_train)
    
    return reg

def studentScore(reg, ages_test, net_worths_test):
    return reg.score(ages_test, net_worths_test)
    


# ## Using the classifier

# In[11]:

#!/usr/bin/python

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image
from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

reg = studentReg(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())

