
# coding: utf-8

# In[7]:

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'UNC played duck in basketball',
    'Duk lost the basketball game',
    'I ate a sandwich'
]

vectorizer = CountVectorizer()
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_


# ### Using stop words
# These words, called stop words, include determiners such as the , a , and
# an ; auxiliary verbs such as do , be , and will ; and prepositions such as on , around , and
# beneath .Stop words are often functional words that contribute to the document's
# meaning through grammar rather than their denotations.

# In[11]:

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'UNC played duck in basketball',
    'Duk lost the basketball game',
    'I ate a sandwich'
]

vectorizer = CountVectorizer(stop_words = 'english')
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
print 

stop_words =  vectorizer.get_stop_words()
print stop_words
print
vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names()
print len(feature_names)


# In[12]:

import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib as plt
import skimage.io as io
from skimage.exposure import equalize_hist


# In[ ]:

#!/usr/bin/python
#http://www.nltk.org/howto/stem.html

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            print path
            ### only look at first 200 emails when developing
            ### once everything is working, remove this line to run over full dataset

            temp_counter += 1

            path = os.path.join('..', path[:-1])

            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            stem_text = parseOutText(email)

            ### use str.replace() to remove any instances of the words ["sara", "shackleton", "chris", "germani"]
            rep = ["sara", "shackleton", "chris", "germani"]
            for item in rep:
                stem_text = stem_text.replace(item,"")

                ### append the text to word_data
                word_data.append(''.join(stem_text))

                ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris

                if name == "sara":
                    from_data.append(0)
                elif name == "chris":
                    from_data.append(1)

            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

#print
#print word_data[152]

### in Part 4, do TfIdf vectorization here

#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#from nltk.corpus import stopwords
#sw = stopwords.words("english")

#transformer = TfidfVectorizer(stop_words = 'english')
transformer = CountVectorizer(stop_words = 'english')

print "fitting the data..."
stop_words =  transformer.get_stop_words()
print
print "No stop words: " + str(len(stop_words))
print

transformer.fit_transform(word_data)

print "getting the features name..."
feature_names = transformer.get_feature_names()
print len(feature_names)
print
print feature_names[34597]
#38757

