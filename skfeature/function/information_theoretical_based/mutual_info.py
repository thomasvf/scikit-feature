from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
import numpy as np
from joblib import Memory


class MutualInformation(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Simple univariate mutual information feature selector."""
    def __init__(self, n_features_to_select=0.05, memory=None, random_state=None):
        """Initialize mRMR.

        Parameters
        ----------
        n_features_to_select : int or float
            Number of features to select.
        memory : Memory
            Memory for caching the results of mutual information scores.
        """
        self.n_features_to_select = n_features_to_select
        self.memory = memory
        self.random_state = random_state

    def fit(self, X, y):
        """Compute the mutual information between features and target.

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
        rng = check_random_state(self.random_state)

        if isinstance(self.n_features_to_select, float):  # todo modify this to a check
            n_features_to_select = round(self.n_features_to_select * n_features)
        else:
            n_features_to_select = self.n_features_to_select

        discrete_target = True
        try:
            check_classification_targets(y)
        except ValueError:
            discrete_target = False

        if discrete_target:
            if self.memory is not None:
                mutual_info_ = self.memory.cache(mutual_info_classif)
                self.scores_ = mutual_info_(X, y, random_state=rng)
            else:
                self.scores_ = mutual_info_classif(X, y, random_state=rng)
        else:
            if self.memory is not None:
                mutual_info_ = self.memory.cache(mutual_info_regression)
                self.scores_ = mutual_info_(X, y, random_state=rng)
            else:
                self.scores_ = mutual_info_regression(X, y, random_state=rng)

        indices = np.argsort(-self.scores_, 0)[:n_features_to_select]
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[indices] = True
        return self

    def _get_support_mask(self):
        return self.support_
