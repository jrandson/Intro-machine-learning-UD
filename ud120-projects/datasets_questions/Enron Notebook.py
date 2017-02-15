
# coding: utf-8

# In[8]:

def show_dic_values(dic):
    for key in dic.keys():
        print key,': ', dic[key]


# In[48]:

#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
names = enron_data.keys()


name = enron_data.keys()[0]
key_ = 'salary'
for key in enron_data.keys():
    if not enron_data[key]['bonus'] == 'NaN' and not enron_data[key]['salary'] == 'NaN' and key != 'TOTAL':
        if enron_data[key]['bonus'] + enron_data[key]['salary'] > enron_data[name]['bonus'] + enron_data[name]['salary']:
            name = key    

print
print "name:", name
print "bonus", enron_data[name]['bonus']
print "Salary", enron_data[name]['salary']
print "--"


# In[10]:

from poi_email_addresses import poiEmails

poi_names_file = open("../final_project/poi_names.txt", "r")
poi_emails = poiEmails()

for e in poi_emails:
    pass#print e
print len(poi_emails)



poi_names = []
for name in poi_names_file:
    if line :       
        poi_names.append(line)


# In[ ]:

poi_name_email = []

for name in poi_names:
    name_ = name.split(',')
    for email in poi_emails:
        print name_[0], name[1]
        if name_[0] in email and name_[1] in email:
            poi_name_email.append((name, email))
            print name, email        
    poi_names.append(line)
        


# In[ ]:



