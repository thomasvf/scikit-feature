from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import KBinsDiscretizer


def discretize(X, y, n_bins):
    d = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X = d.fit_transform(X, y)
    return X


def check_need_discrete_features(obj):
    """Check if given instance of SelectorMixin needs discrete features."""
    assert isinstance(obj, SelectorMixin)

    if hasattr(obj, 'n_bins'):  # todo: generalize the way to pass a discretizer to the FS methods and then change this
        return True
    return False
