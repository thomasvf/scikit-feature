
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin, f_classif

import numpy as np
from joblib import Memory


def f_score(X, y):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn),
    where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features,)
        f-score for each feature
    """

    pt = PowerTransformer()
    X = pt.fit_transform(X)
    F, pval = f_classif(X, y)
    return F


def feature_ranking(F):
    """
    Rank features in descending order according to t-score, the higher the t-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1]


class FTest(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
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
        Fit the Ftest.
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
            ftest = self.memory.cache(f_score)
            scores_ = ftest(X, y)
        else:
            scores_ = f_score(X, y)

        indices = feature_ranking(scores_)[:n_features_to_select]

        self.scores_ = scores_
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
