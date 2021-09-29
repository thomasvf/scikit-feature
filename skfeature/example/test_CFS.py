import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from skfeature.function.statistical_based import CFS
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    # ss = StratifiedKFold(n_samples, n_folds=10, shuffle=True)
    n_splits = 2
    ss = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # perform evaluation on classification task
    num_fea = 100    # number of selected features
    clf = svm.LinearSVC()    # linear SVM
    fs = CFS.CFS()

    pipe = Pipeline([
        ('fs', fs),
        ('clf', clf)
    ])

    reports = {'f1_score': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'support_indices': []}
    for r, (train, test) in enumerate(ss.split(X, y)):
        print("Run [%d/%d]" % (r+1, n_splits))
        pipe = pipe.fit(X[train], y[train])
        y_pred = pipe.predict(X[test])

        reports['f1_score'].append(f1_score(y[test], y_pred))
        reports['accuracy_score'].append(accuracy_score(y[test], y_pred))
        reports['recall_score'].append(recall_score(y[test], y_pred))
        reports['precision_score'].append(precision_score(y[test], y_pred))
        reports['support_indices'].append(pipe.named_steps['fs'].get_support(indices=True))

    print(reports)
    f1_scores = pd.Series(reports['f1_score'])
    f1_scores.plot.hist(bins=5)

    #
    # correct = 0
    # for train, test in ss.split(X, y):
    #     # obtain the index of selected features on training set
    #     idx = CFS.cfs(X[train], y[train])
    #
    #     # obtain the dataset on the selected features
    #     selected_features = X[:, idx[0:num_fea]]
    #
    #     # train a classification model with the selected features on the training dataset
    #     clf.fit(selected_features[train], y[train])
    #
    #     # predict the class labels of test data
    #     y_predict = clf.predict(selected_features[test])
    #
    #     # obtain the classification accuracy on the test data
    #     acc = accuracy_score(y[test], y_predict)
    #     correct = correct + acc
    #
    # # output the average classification accuracy over all 10 folds
    # print('Accuracy:', float(correct) / 10)


if __name__ == '__main__':
    main()
