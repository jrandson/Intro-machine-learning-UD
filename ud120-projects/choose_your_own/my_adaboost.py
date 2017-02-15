#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

from sklearn.ensemble  import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

def accuracy(prediction, labels_test):
    rate = 0
    for i in range(len(labels_test)):
        if prediction[i] == labels_test[i]:
            rate += 1

    return 1.0*rate/len(labels_test)

decision_tree_clf = DecisionTreeClassifier(min_samples_split=10, max_depth = 1)
clf  =  AdaBoostClassifier(decision_tree_clf, algorithm='SAMME',n_estimators=200)

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

acc = accuracy(prediction, labels_test)

try:
    print "Accuracy: ", acc
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
