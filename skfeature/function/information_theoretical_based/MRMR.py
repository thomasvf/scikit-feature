from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

from skfeature.function.information_theoretical_based import LCSI
from skfeature.utility.discretization import discretize
from joblib.memory import Memory


def mrmr(X, y, **kwargs):
    """
    This function implements the MRMR feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR')
    return F, J_CMI, MIfy


class MinimumRedundancyMaximumRelevance(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Minimum Redundancy Maximum Relevance feature selection algorithm."""
    def __init__(self, n_features_to_select=None, memory=None, n_bins=None, max_features=None):
        """Initialize mRMR.

        Parameters
        ----------
        n_features_to_select : int
            Number of features to select from the features returned.
        memory : Memory
            Memory to cache mrmr execution.
        n_bins : int
            Number of bins to use in the discretization. If None, then discretization is not applied.
        max_features : int
            If given, forces algorithm to look for max_features.
        """
        self.n_features_to_select = n_features_to_select
        self.memory = memory
        self.n_bins = n_bins
        self.max_features = max_features

    def fit(self, X, y):
        """
        Fit the mRMR.

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

        if isinstance(self.n_features_to_select, float):  # todo modify this to a check
            n_features_to_select = round(self.n_features_to_select * n_features)
        else:
            n_features_to_select = self.n_features_to_select

        if self.n_bins is not None:
            if self.memory is not None:
                discretize_ = self.memory.cache(discretize)
                X = discretize_(X, y, self.n_bins)
            else:
                X = discretize(X, y, self.n_bins)

        if self.memory is not None:
            mrmr_ = self.memory.cache(mrmr)
            if self.max_features:
                indices, j_cmi, mi_y = mrmr_(X, y, n_selected_features=self.max_features)
            else:
                indices, j_cmi, mi_y = mrmr_(X, y)
        else:
            if self.max_features:
                indices, j_cmi, mi_y = mrmr(X, y, n_selected_features=self.max_features)
            else:
                indices, j_cmi, mi_y = mrmr(X, y)

        self.scores_ = np.zeros(n_features, dtype=float)
        self.scores_[indices] = j_cmi

        self.mi_y_ = np.zeros(n_features, dtype=float)
        self.mi_y_[indices] = mi_y

        if n_features_to_select:
            indices = indices[:n_features_to_select]

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
