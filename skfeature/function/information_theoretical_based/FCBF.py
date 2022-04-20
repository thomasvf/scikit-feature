import numpy as np
from skfeature.utility.mutual_information import su_calculation
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from skfeature.utility.discretization import discretize


def fcbf(X, y, **kwargs):
    """
    This function implements Fast Correlation Based Filter algorithm

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        delta: {float}
            delta is a threshold parameter, the default value of delta is 0

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    SU: {numpy array}, shape (n_features,)
        symmetrical uncertainty of selected features

    Reference
    ---------
        Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
    """
    print('fcbf')
    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0

    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
    t1 = np.zeros((n_features, 2))
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)

    # s_list = (t1[t1[:, 1] > delta, :]).astype(np.int)
    s_list = (t1[t1[:, 1] > delta, :])
    # index of selected features, initialized to be empty
    F = []
    # Symmetrical uncertainty of selected features
    SU = []
    while len(s_list) != 0:
        # select the largest su inside s_list
        idx = np.argmax(s_list[:, 1])
        # record the index of the feature with the largest su
        fp = X[:, int(s_list[idx, 0])]
        np.delete(s_list, idx, 0)

        F.append(s_list[idx, 0])
        SU.append(s_list[idx, 1])
        for i in s_list[:, 0]:
            i_int = int(i)
            fi = X[:, i_int]
            if su_calculation(fp, fi) >= t1[i_int, 1]:
                # construct the mask for feature whose su is larger than su(fp,y)
                idx = s_list[:, 0] != i_int
                idx = np.array([idx, idx])
                idx = np.transpose(idx)
                # delete the feature by using the mask
                s_list = s_list[idx]
                length = len(s_list)//2
                s_list = s_list.reshape((length, 2))
    return np.array(F, dtype=int), np.array(SU)


class FastCorrelationBasedFilter(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Fast Correlation Based Filter algorithm."""
    def __init__(self, delta=0, n_features_to_select=None, memory=None, n_bins=None):
        """Initialize Fast Correlation based Filter.

        Parameters
        ----------
        delta : float
            Threshold parameter. #todo threshold for what?
        n_features_to_select : int
            Number of features to use.
        memory : Memory
            Memory for caching the results of the FCBF.
        n_bins : int
            Number of bins to use in the discretization. If None, then discretization is not applied.
        """
        self.delta = delta
        self.n_features_to_select = n_features_to_select
        self.memory = memory
        self.n_bins = n_bins

    def fit(self, X, y):
        """
        Fit the FCBF.

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
        if self.n_bins is not None:
            if self.memory is not None:
                discretize_ = self.memory.cache(discretize)
                X = discretize_(X, y, self.n_bins)
            else:
                print("discretize")
                X = discretize(X, y, self.n_bins)

        if self.memory is not None:
            fcbf_ = self.memory.cache(fcbf)
            indices, su = fcbf_(X, y)
        else:
            indices, su = fcbf(X, y)

        self.scores_ = np.zeros(n_features, dtype=float)
        self.scores_[indices] = su

        if self.n_features_to_select is not None:
            indices = indices[:self.n_features_to_select]

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True

        return self

    def _get_support_mask(self):
        return self.support_
