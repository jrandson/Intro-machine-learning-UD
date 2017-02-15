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

import sys
sys.path.insert(0, '../final_project/')
from poi_email_addresses import poiEmails
import pickle
import numpy as np

def show_data_by_name(name):
    for key in enron_data[name]:
        print key,":", enron_data[name][key]

def get_poi_list(poi_emails, enron_data):
    poi_list = {}
    for name in enron_data.keys():
        poi = enron_data[name]
        if poi[name]['email_addres'] in poi_emails:
            poi_list.set_default(name, poi['email_addres'])

    return poi_list

def load_poi_names():
    poi_names_file = open("../final_project/poi_names.txt", "r")
    poi_names = []
    for line in poi_names_file:
        if '(y)' in line or '(n)' in line:
            poi_names.append(line.strip())

    return poi_names

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_names = load_poi_names()
poi_emails =  poiEmails()

def get_stock(name):
    name = name.upper().strip()
    return enron_data[name.upper()]['total_stock_value']

def get_keys():
    dic = enron_data[enron_data.keys()[0]].keys()
    dic.sort()
    return dic

def get_total_email_messages(name):
    name = name.upper().strip()
    return enron_data[name]['from_this_person_to_poi']

def get_stock_options(name):
    name = name.upper().strip()
    return enron_data[name]['exercised_stock_options']

def get_by_last_name(last_name):
    for name in enron_data.keys():
        if last_name.upper().strip() in name:
            return  enron_data[name]

    return ""

print get_keys()

def comparing_payments():
    data = {'lay': get_by_last_name('lay'), 'skilling': get_by_last_name('skilling'),
            'fastow': get_by_last_name('fastow')}

    for key in data:
        print key, data[key]['total_payments'], data[key]['deferral_payments'], data[key]['director_fees'], data[key]['salary']

def values_blank():
    names = enron_data.keys()

    quantfied_salary = []
    quantified_email_address= []
    for name in names:
        if not enron_data[name]['salary'] == 'NaN':
            quantfied_salary.append(enron_data[name])
        if not enron_data[name]['email_address']  == 'NaN':
            quantified_email_address.append(enron_data[name]['email_address'])

    print 'quantify salary', len(quantfied_salary)
    print 'email address', len(quantified_email_address)

def get_blank_payments():
    payments_blank = []
    total = 0
    for name in names:
        if enron_data[name]['poi']:
            total += 1
            if enron_data[name]['total_payments'] == 'NaN':
                payments_blank.append(enron_data[name])

    print 'payments blank', 100.0*len(payments_blank)/total , '%'
    print 'total', total



def add_data(data):
    new_entry = data.values()[0]
    new_entry['poi'] = True
    new_entry['total_payments'] = 'NaN'

    new_data = data
    for i in range(10):
        new_data['No Name_'+str(i)] = new_entry

    return new_data

def get_blank_payments2():
    enron_data_new = add_data(enron_data)
    names = enron_data_new.keys()
    print names
    payments_blank = []
    total_poi = 0
    for name in names:
        if enron_data[name]['poi']:
            total_poi += 1
            if enron_data[name]['total_payments'] == 'NaN' :
                payments_blank.append(enron_data[name])


    num_payments_blank = len(payments_blank)
    print
    print 'payments blank', num_payments_blank , '%'
    print 'total', total_poi

get_blank_payments2()
#show_data_by_name(names[0])
