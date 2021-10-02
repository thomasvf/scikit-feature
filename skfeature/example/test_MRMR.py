import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from skfeature.function.information_theoretical_based import MRMR
from joblib.memory import Memory


def test_workings():
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
    y = y[:, 0]
    n_samples, n_features = X.shape  # number of samples and number of features
    print("#Samples: %d #Features: %d" % (n_samples, n_features))

    memory = Memory('./cachedir')
    fs = MRMR.MinimumRedundancyMaximumRelevance(n_features_to_select=100, memory=memory)
    fs.fit(X, y)
    print(fs.get_support(indices=True))


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
    n_repeats = 2
    ss = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=0)

    # perform evaluation on classification task
    num_fea = 10  # number of selected features
    clf = svm.SVC()  # linear SVM
    memory = Memory('./cachedir')
    fs = MRMR.MinimumRedundancyMaximumRelevance(n_features_to_select=num_fea, memory=memory)
    pipe = Pipeline([
        ('fs', fs),
        ('clf', clf)
    ])
    params = [
        {'fs__n_features_to_select': [1, 2, 3, 4, 5, 10, 50, 100, 500], 'clf__kernel': ['linear'],
         'clf__C': [0.3, 3, 30, 300]},
        {'fs__n_features_to_select': [1, 2, 3, 4, 5, 10, 50, 100, 500], 'clf__kernel': ['rbf'],
         'clf__C': [0.3, 3, 30, 300], 'clf__gamma': [0.0003, 0.003, 0.03, 0.3]},
    ]

    reports = {'f1_scores': [], 'accuracy_scores': [], 'precision_scores': [], 'recall_scores': [],
               'support_indices': [], 'best_params': []}
    for r, (train, test) in enumerate(ss.split(X, y)):
        print("Run [%d/%d]" % (r + 1, n_splits*n_repeats))
        grid_search = GridSearchCV(pipe, param_grid=params, scoring='f1', refit=True, verbose=10)
        grid_search.fit(X[train], y[train])
        y_pred = grid_search.predict(X[test])

        reports['f1_scores'].append(f1_score(y[test], y_pred))
        reports['accuracy_scores'].append(accuracy_score(y[test], y_pred))
        reports['recall_scores'].append(recall_score(y[test], y_pred))
        reports['precision_scores'].append(precision_score(y[test], y_pred))
        reports['support_indices'].append(grid_search.best_estimator_.named_steps['fs'].get_support(indices=True))
        reports['best_params'].append(grid_search.best_params_)

    f1_scores = pd.Series(reports['f1_scores'])
    accuracy_scores = pd.Series(reports['accuracy_scores'])
    recall_scores = pd.Series(reports['recall_scores'])
    precision_scores = pd.Series(reports['precision_scores'])
    all_selected_features = np.array([], dtype=np.int)
    for selected_features in reports['support_indices']:
        all_selected_features = np.concatenate((all_selected_features, selected_features))
    n_times_selected = pd.Series(np.bincount(all_selected_features) / (n_repeats * n_splits))
    n_times_selected.sort_values(ascending=False, inplace=True)

    print("BEST PARAMS")
    print(reports['best_params'])
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
    memory.clear()


if __name__ == '__main__':
    main()
