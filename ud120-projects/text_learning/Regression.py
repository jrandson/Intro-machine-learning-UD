
# coding: utf-8

# In[11]:

import numpy as np
import matplotlib.pyplot as plt

# y  = b*a^x
b = 0.048
a = 1.29
x = range(30)

y = [b*pow(a,i) + 10*np.random.rand() for i in x]

plt.scatter(x,y)
plt.show()


# #### y = b*a^x 
# #### ln(y) = ln (b*a^x)
# #### ln(y) = ln(a)*x + ln(b)
# ###  Y = Ax + B
# 
# #### Y = ln(y)
# #### A = ln(a)
# #### B = ln(b)
# 

# In[39]:

Y = np.log(y)

n = sum(x)
sum_x = sum(x)
sum_x2 = sum([ i**2 for i in x])
sum_xy = sum([i[0]*i[1] for i in zip(x,Y)])

MV = np.array([[n, sum_x],[sum_x, sum_x2]])
y_ = np.array([[sum_x],[sum_xy]])

print x_
print MV.dot(x_)

