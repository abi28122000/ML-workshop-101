#!/usr/bin/python

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('gtk')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def prettyPicture(clf, X_test, y_test, fileName):

    """
    Method to obtain the image of the classifier and the decision surface
    :param clf: classifier to be used on the test data
    :param X_test: testing data containing the features from data set
    :param y_test: actual labels assigned to the test data features
    :param fileName: name of the file to save the plot in
    :return:
    """

    ### defining the min max range of each of the axes in the plot
    x_min = 0.0;
    x_max = 1.0
    y_min = 0.0;
    y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points to visualize the accuracy of the decision boundary
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 1]

    plt.scatter(grade_sig, bumpy_sig, color="b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color="r", label="slow")

    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig(fileName)
    plt.show()
    return