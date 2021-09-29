import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from skfeature.function.similarity_based import reliefF



def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features
    print("#Samples: %d #Features: %d" % (n_samples, n_features))

    n_splits = 2
    n_repeats = 50
    ss = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=0)

    # perform evaluation on classification task
    num_fea = 20  # number of selected features
    clf = svm.LinearSVC()  # linear SVM
    fs = reliefF.ReliefF(n_features_to_select=10)
    pipe = Pipeline([
        ('fs', fs),
        ('clf', clf)
    ])

    reports = {'f1_scores': [], 'accuracy_scores': [], 'precision_scores': [], 'recall_scores': [], 'support_indices': []}
    for r, (train, test) in enumerate(ss.split(X, y)):
        print("Run [%d/%d]" % (r + 1, n_splits*n_repeats))
        pipe = pipe.fit(X[train], y[train])
        y_pred = pipe.predict(X[test])

        reports['f1_scores'].append(f1_score(y[test], y_pred))
        reports['accuracy_scores'].append(accuracy_score(y[test], y_pred))
        reports['recall_scores'].append(recall_score(y[test], y_pred))
        reports['precision_scores'].append(precision_score(y[test], y_pred))
        reports['support_indices'].append(pipe.named_steps['fs'].get_support(indices=True))

    f1_scores = pd.Series(reports['f1_scores'])
    accuracy_scores = pd.Series(reports['accuracy_scores'])
    recall_scores = pd.Series(reports['recall_scores'])
    precision_scores = pd.Series(reports['precision_scores'])
    all_selected_features = np.array([], dtype=np.int)
    for selected_features in reports['support_indices']:
        all_selected_features = np.concatenate((all_selected_features, selected_features))
    n_times_selected = pd.Series(np.bincount(all_selected_features) / (n_repeats * n_splits))
    n_times_selected.sort_values(ascending=False, inplace=True)

    print(f1_scores.describe())
    print(accuracy_scores.describe())
    print(recall_scores.describe())
    print(precision_scores.describe())
    print(n_times_selected.describe())

    fig, ax = plt.subplots(2, 2)
    accuracy_scores.plot.hist(bins=10, ax=ax[0, 0])
    f1_scores.plot.hist(bins=10, ax=ax[0, 1])
    precision_scores.plot.hist(bins=10, ax=ax[1, 0])
    recall_scores.plot.hist(bins=10, ax=ax[1, 1])

    fig, ax = plt.subplots()
    n_times_selected[:20].plot.bar(ax=ax)
    plt.show()


# def main():
#     # load data
#     mat = scipy.io.loadmat('../data/COIL20.mat')
#     X = mat['X']    # data
#     X = X.astype(float)
#     y = mat['Y']    # label
#     y = y[:, 0]
#     n_samples, n_features = X.shape    # number of samples and number of features
#
#     # split data into 10 folds
#     ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
#
#     # perform evaluation on classification task
#     num_fea = 100    # number of selected features
#     clf = svm.LinearSVC()    # linear SVM
#
#     correct = 0
#     for train, test in ss:
#         # obtain the score of each feature on the training set
#         score = reliefF.reliefF(X[train], y[train])
#
#         # rank features in descending order according to score
#         idx = reliefF.feature_ranking(score)
#
#         # obtain the dataset on the selected features
#         selected_features = X[:, idx[0:num_fea]]
#
#         # train a classification model with the selected features on the training dataset
#         clf.fit(selected_features[train], y[train])
#
#         # predict the class labels of test data
#         y_predict = clf.predict(selected_features[test])
#
#         # obtain the classification accuracy on the test data
#         acc = accuracy_score(y[test], y_predict)
#         correct = correct + acc
#
#     # output the average classification accuracy over all 10 folds
#     print 'Accuracy:', float(correct)/10

if __name__ == '__main__':
    main()