
# coding: utf-8

# In[26]:

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

str1 = "Hi Kate the self drive car will be the late best Sebastian"
str2 = "Hi Sebastian the machine learning class will be great great great katie"
str3 = "Hi kate the machine learn class will be most excellent"

email_list = [str1, str2, str3]

bag_of_words = vectorizer.fit(email_list) #fit_transform(email_list)
bag_of_words = vectorizer.transform(email_list)

list_word = str1.split(' ') + str2.split(' ') + str3.split(' ')

tmp = []
for w in list_word:
    if not w in tmp:
        print w, vectorizer.vocabulary_.get(w)
        tmp.append(w)

#print vectorizer.vocabulary_.get("great")



# In[27]:

from nltk.corpus import stopwords

#identifica as palvras stop words, aquelas que não são relevantes apara o processo de treinamento do algoritmo.
#Em geral são palavras que aparecem em todos os textos, como modais, conectores, preposições, etc
sw = stopwords.words("english")

#use  nltk.download() to download de dictionary of words

print len(sw)


# In[1]:

from nltk.stem.porter import *

#changin to single
stemmer = PorterStemmer()

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized', 
           'meeting', 'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference', 
           'colonizer','plotted']

single = [stemmer.stem(plural) for plural in plurals]

#print single


#create a new instance of language
stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

print stemmer.stem("running")
print stemmer2.stem("having")

stemmer = SnowballStemmer("portuguese")
print stemmer.stem("felicidade")


# In[12]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")


# In[ ]:

#!/usr/bin/python
#http://www.nltk.org/howto/stem.html

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string               

        ### split the text string into individual words, stem each word
        
        stemmer = SnowballStemmer("english")
        
        word_stem_list = []
        for item in text_string.split(' '):
            if len(item) > 0:
                word_stem_list.append(stemmer.stem(item))
        
        ### and append the stemmed word to words (make sure there's a single          
        ### space between each stemmed word)
        words = ' '.join(word_stem_list)      
        

    return words

def main():
    ff = open("../text_learning/test_email.txt", "r")
        
    text = parseOutText(ff)
    print text

if __name__ == '__main__':
    main()



# In[11]:

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
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        
        temp_counter += 1        
        if temp_counter < 200:
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

#print word_data[152]

### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#from nltk.corpus import stopwords
#sw = stopwords.words("english")
     
transformer = TfidfVectorizer(stop_words = 'english')
vectorizer = CountVectorizer(stop_words = 'english')

print
stop_words =  transformer.get_stop_words()
print "count stop words Tfid", len(stop_words)

stop_words =  transformer.get_stop_words()
print "count stop words Vectorize", len(stop_words)

transformer.fit_transform(word_data)
vectorizer.fit_transform(word_data)

feature_names = transformer.get_feature_names()
print len(feature_names)

feature_names = vectorizer.get_feature_names()
print len(feature_names)



# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer
     

transformer = TfidfVectorizer()

#Remove stopwords
#stop_words =  transformer.get_stop_words()
#print stop_words

transformer.fit_transform(word_data)

feature_names = transformer.get_feature_names()
print len(feature_names)


