### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):

    """
    Method to fit a classifier on the training data and return the classifier
    :param features_train: training feature set obtained from the actual data
    :param labels_train: classified labels corresponding to the training data
    :return: gaussianNB classifier and fileName to store the visualization
    """

    ### create the classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### file name to store the resulting visualization
    fileName = "./visualizations/nb/gaussianNB.png"

    ### return the classifier and the fileName to store the visualization
    return clf,fileName