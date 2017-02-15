#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

# kernel = [linear, rbf, poly]

clf_rbf = svm.SVC(kernel='rbf', C = 10000)
clf_poly = svm.SVC(kernel='poly', C=1e3, degree=8)
clf_linear = svm.SVC(kernel='linear',C=1e3)

clf = clf_rbf

features_train_lim = features_train[:len(features_train)/100]
labels_train_lim = labels_train[:len(labels_train)/100]

t = time()
clf.fit(features_train, labels_train)
#clf_linear.fit(features_train, labels_train)
t_train = time() - t

t = time()
prediction = clf.predict(features_test)
t_predict = time() - t
prediction_ = [np.int(round(i)) for i in prediction]

rate = 0
sara_emails = 0
chris_emails = 0
for i in range(len(prediction)):
    if prediction[i] == 0:
        sara_emails += 1
    elif prediction[i] == 1:
        chris_emails += 1

    if prediction_[i] == labels_test[i]:
        rate += 1
accuracy = np.float(rate) / len(prediction)

#accuracy = accuracy_score(labels_test, prediction_)
print "Accuracy: ", accuracy
print "train time:", round(t_train,2), " s"
print "predict time:", round(t_predict,2), " s"

print ""
print "Emails for Sara: ", sara_emails
print "Emails for Chris: ", chris_emails
print ""

names = ['Sara', 'Chris']
print "The prediction of element 10 gives the result: ", names[prediction[10]]
print "The prediction of element 26 gives the result: ", names[prediction[26]]
print "The prediction of element 50 gives the result: ", names[prediction[50]]

#########################################################
