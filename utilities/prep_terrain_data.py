#!/usr/bin/python
import random


def makeTerrainData(n_points=1000):

    """
    Method to create the dummy Terrain data set.
    :param n_points: Number of sample points to be created.
    :return: trainingFeatures, trainingLabels, testFeatures, testLabels
    """

    ### creates a random number generator to generate numbers between 0,1
    random.seed(42)

    ### generating random terrain data for grade and bumpiness
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]

    ### random error in the toy data set is assumed
    error = [random.random() for ii in range(0,n_points)]

    ### error calculation formula and error adjustment
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

    ### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    ### returning the training and test data that has been created
    return X_train, y_train, X_test, y_test