### import the sklearn module for SVM
from sklearn.svm import SVC

def classify(features_train, labels_train, kernel="rbf", gamma="auto", cValue=1.0):

    """
    Method to fit the classifier on the training data and return the classifier
    :param features_train: training feature set obtained from the actual data
    :param labels_train: classified labels corresponding to the training data
    :param kernel: transformation which should be applied (rbf, poly, auto)
    :param gamma: the kernel coefficient value determining the width of the kernel
    :param cValue: larger the value more is the importance given to classifying the
                   data points correctly over choosing a larger margin hyperplane
    :return: SVM classifier and the file name to store the visualization
    """

    ### create the classifier
    clf = SVC(kernel=kernel,gamma=gamma,C=cValue)

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### file name to store the resulting visualization
    fileName = "./visualizations/svm/svmSVC-"+kernel+"-"+str(gamma)+"-"+str(cValue)+".png"

    ### return the classifier and the fileName to store the visualization
    return clf,fileName