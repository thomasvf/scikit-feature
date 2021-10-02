from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
import numpy as np

from skfeature.function.information_theoretical_based import LCSI


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
    def __init__(self, n_features_to_select=None):
        """Initialize mRMR.

        Parameters
        ----------
        n_features_to_select : int
            Number of features to select.
        """
        self.n_features_to_select = n_features_to_select

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
        if self.n_features_to_select:
            indices, self.j_cmi_, self.mi_fy_ = mrmr(X, y, n_selected_features=self.n_features_to_select)
        else:
            indices, self.j_cmi_, self.mi_fy_ = mrmr(X, y, n_selected_features=self.n_features_to_select)

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
