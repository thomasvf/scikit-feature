import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based.FCBF import FastCorrelationBasedFilter
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np

from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based.mutual_info import MutualInformation

import joblib


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
    n_repeats = 1
    ss = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=0)

    # perform evaluation on classification task
    num_fea = 100  # number of selected features
    params = [
        {'fs__n_features_to_select': [1, 2, 3, 4, 5, 10, 50, 100, 500], 'clf__kernel': ['linear'],
         'clf__C': [0.3, 3, 30, 300]},
        {'fs__n_features_to_select': [1, 2, 3, 4, 5, 10, 50, 100, 500], 'clf__kernel': ['rbf'],
         'clf__C': [0.3, 3, 30, 300], 'clf__gamma': [0.0003, 0.003, 0.03, 0.3]},
    ]
    memory = joblib.Memory('./cachedir')
    clf = svm.SVC()  # linear SVM
    fs = FCBF.FastCorrelationBasedFilter(delta=0, memory=memory)

    pipe = Pipeline([
        ('fs', fs),
        ('clf', clf)
    ])

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
    memory.clear()


from typing import Union, Tuple, Dict
path_to_csv = "/home/thomas/work/projects/ml/Gene Selection Tireoide/data/CancerTireoide_ClassifBin.csv"


def load_csv_features_target(path_to_csv: str, return_X_y: bool = False, return_df: bool = False) -> \
        Union[Dict, Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
    """Load a basic csv dataset.

    The first row of the dataset defines the name of each column. Each column must correspond to an attribute.
    The last column is the target.

    If return_X_y is false, then it returns a dictionary containing the fields data, target, feature_names and
    target_name.
    The names are inferred from the first row of the csv file.

    :return: the dataset
    """
    df = pd.read_csv(path_to_csv, index_col=[0])
    df = df.drop(columns=['geo_accession'])
    df['DiagnosisDi'].loc[df['DiagnosisDi'] == 'Cancer'] = 1
    df['DiagnosisDi'].loc[df['DiagnosisDi'] == 'Not Cancer'] = 0
    df['DiagnosisDi'] = df['DiagnosisDi'].astype(np.uint8)

    if return_df:
        return df

    data = df.iloc[:, :-1].to_numpy()
    target = df.iloc[:, -1].to_numpy()
    feature_names = df.columns[:-1]
    target_name = df.columns[-1]

    if return_X_y:
        return data, target

    return {
        'data': data,
        'target': target,
        'feature_names': feature_names,
        'target_name': target_name
    }


def main2():
    from sklearn.feature_selection import VarianceThreshold

    X = np.load('X.npy')
    y = np.load('y.npy')

    X_var = VarianceThreshold(2e-4).fit_transform(X)
    print(X_var.shape)

    print(X.shape)

    fcbf = FastCorrelationBasedFilter(delta=0, n_features_to_select=10, n_bins=5)
    fcbf.fit(X_var, y)
    f_fcbf = fcbf.get_support(indices=True)

    mi = MutualInformation(n_features_to_select=10)
    mi.fit(X_var, y)
    f_mi = mi.get_support(indices=True)

    print(f_fcbf)
    print(f_mi)


if __name__ == '__main__':
    main2()
