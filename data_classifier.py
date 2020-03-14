#!/usr/bin/python

"""
 This is a driver code which calls the classify method
 from the respective classifier included in import.

 The same terrain classifier data can be run through
 different classifiers and the results can be compared.
"""

from utilities.prep_terrain_data import makeTerrainData
from utilities.class_vis import prettyPicture
from classifiers.gaussianNB_classifier import classify
#from classifiers.svm_classifier import classify

import numpy as np
import pylab as pl
import subprocess

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast"
### and "slow" points mixed in together--separate them so we can give
### them different colors in the scatterplot,and visually identify them.

grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

### This method is called from the imported classifier.
clf, fileName = classify(features_train, labels_train)

### call predict method on the classifier to obtain the prediction over test data
#pred = clf.predict(features_test)
### commented as it is called internally while plotting the decision surface

### plot the decision surface as an image
prettyPicture(clf, features_test, labels_test, fileName)

### calculate the accuracy of the classifier and print it
accuracy = clf.score(features_test, labels_test)
print "accuracy :" + str(accuracy)






