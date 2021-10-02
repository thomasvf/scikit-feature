from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
import numpy as np

from skfeature.function.information_theoretical_based import LCSI
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
    def __init__(self, n_features_to_select=None, memory=None):
        """Initialize mRMR.

        Parameters
        ----------
        n_features_to_select : int
            Number of features to select.
        memory : Memory
            Memory to cache mrmr execution.
        """
        self.n_features_to_select = n_features_to_select
        self.memory = memory

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
        if self.memory is not None:
            mrmr_ = self.memory.cache(mrmr)
            indices, _, _ = mrmr_(X, y)
        else:
            indices, _, _ = mrmr(X, y)

        if self.n_features_to_select:
            if self.n_features_to_select > len(indices):
                raise ValueError("Number of features to select ({} features) in MRMR is greater than the number of features "
                      "returned by the algorithm ({} features).".format(self.n_features_to_select, len(indices)))

            indices = indices[:self.n_features_to_select]

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
