#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###


def accuracy(prediction, labels_test):
    rate = 0
    for i in range(len(labels_test)):
        if prediction[i] == labels_test[i]:
            rate += 1

    return 1.0*rate/len(labels_test)

print "size features", len(features_train[0])

min_samples_split = 40
clf_tree = tree.DecisionTreeClassifier(min_samples_split = min_samples_split)

clf_tree.fit(features_train, labels_train)
prediction = clf_tree.predict(features_test)
#prediction_prob = clf_tree.predict_proba(features_test)

print "Accuracy: ", accuracy(prediction, labels_test)






#########################################################
