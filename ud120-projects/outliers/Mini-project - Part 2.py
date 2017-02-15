
# coding: utf-8

# In[4]:


import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# ### read in data dictionary, convert to numpy array

# In[5]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL')
data = featureFormat(data_dict, features)

column = ['bonus','salary']
df = pd.DataFrame(data_dict)
df = df.T

no_bonus_nan = df['bonus'] != 'NaN'
no_salary_nan = df['salary'] != 'NaN'

df = df[column][no_bonus_nan & no_salary_nan]

print df[(df['bonus'] > 5e6) & (df['salary'] > 1e6)]
#print df[column].sort_values(by='bonus', ascending=False)[:10]

plt.scatter(df['bonus'],df['salary'])
plt.show()


#df = pd.DataFrame(data_dict.values())
#df.head()


# ### your code below

# In[ ]:




# In[6]:

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )



matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

