
# coding: utf-8

# In[35]:

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



# In[47]:

from nltk.corpus import stopwords

sw = stopwords.words("english")

#use  nltk.download() to download de dictionary of words

print len(sw)


# In[50]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")

