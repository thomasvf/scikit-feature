import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from skfeature.function.information_theoretical_based import FCBF


def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features
    print("#Samples: %d #Features: %d" % (n_samples, n_features))
    # split data into 10 folds
    # ss = StratifiedKFold(n_samples, n_folds=10, shuffle=True)
    n_splits = 2
    n_repeats = 10
    ss = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=0)

    # perform evaluation on classification task
    num_fea = 100  # number of selected features
    clf = svm.LinearSVC()  # linear SVM
    fs = FCBF.FastCorrelationBasedFilter(delta=0)

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

    fig, ax = plt.subplots(2, 2)
    f1_scores = pd.Series(reports['f1_scores'])
    accuracy_scores = pd.Series(reports['accuracy_scores'])
    recall_scores = pd.Series(reports['recall_scores'])
    precision_scores = pd.Series(reports['precision_scores'])
    print(f1_scores.describe())
    print(accuracy_scores.describe())
    print(recall_scores.describe())
    print(precision_scores.describe())

    accuracy_scores.plot.hist(bins=10, ax=ax[0, 0])
    f1_scores.plot.hist(bins=10, ax=ax[0, 1])
    precision_scores.plot.hist(bins=10, ax=ax[1, 0])
    recall_scores.plot.hist(bins=10, ax=ax[1, 1])
    plt.show()


if __name__ == '__main__':
    main()
