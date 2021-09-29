import numpy as np
from skfeature.utility.mutual_information import su_calculation
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin


def merit_calculation(X, y):
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X

    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ----------
    merits: {float}
        merit of a feature subset X
    """

    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    """
    This function uses a correlation based heuristic to evaluate the worth of features which is called CFS

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}
        index of selected features

    Reference
    ---------
    Zhao, Zheng et al. "Advancing Feature Selection Research - ASU Feature Selection Repository" 2010.
    """

    n_samples, n_features = X.shape
    F = []
    # M stores the merit values
    M = []
    while True:
        print("Iterate...")
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break

    return np.array(F, dtype=int)


class CFS(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Correlation based heuristic to evaluate the worth of features which is called CFS."""
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the CFS.

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
        indices = cfs(X, y)
        self._support = np.zeros(n_features, dtype=bool)
        self._support[indices] = True

        return self

    def _get_support_mask(self):
        return self._support
