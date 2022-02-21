import pandas as pd

from scipy import stats

from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin

import numpy as np
from joblib import Memory


def t_score(X, y):
    """
    This function calculates t_score for each feature, where t_score is only used for binary problem
    t_score = |mean1-mean2|/sqrt(((std1^2)/n1)+((std2^2)/n2)))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features,)
        t-score for each feature
    """

    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    c = np.unique(y)
    pt = PowerTransformer()
    X = pt.fit_transform(X)
    if len(c) == 2:
        for i in range(n_features):
            f = X[:, i]
            # class0 contains instances belonging to the first class
            # class1 contains instances belonging to the second class
            class0 = f[y == c[0]]
            class1 = f[y == c[1]]
            F[i], pval = stats.ttest_ind(class0, class1, equal_var=False)
    else:
        print('y should be guaranteed to a binary class vector')
        exit(0)
    return np.abs(F)


def feature_ranking(F):
    """
    Rank features in descending order according to t-score, the higher the t-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1]


class TTest(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """ReliefF feature selection algorithm."""

    def __init__(self, n_features_to_select=0.05, memory=None):
        """Initialize mRMR.
        Parameters
        ----------
        n_features_to_select : int or float
            Number of features to select.
        memory : Memory
            Memory for caching the results of reliefF.
        """
        self.n_features_to_select = n_features_to_select
        self.memory = memory

    def fit(self, X, y):
        """
        Fit the Ttest.
        Parameters
        ----------
        X : {np.ndarray} of shape (n_samples, n_features)
            The training samples.
        y : {numpy array} of shape (n_samples,)
            The training labels
        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_features = X.shape[1]
        if isinstance(self.n_features_to_select, float):
            n_features_to_select = round(self.n_features_to_select * n_features)
        else:
            n_features_to_select = self.n_features_to_select

        if self.memory is not None:
            ttest = self.memory.cache(t_score)
            scores_ = ttest(X, y)
        else:
            scores_ = t_score(X, y)

        indices = feature_ranking(scores_)[:n_features_to_select]

        self.scores_ = scores_
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
