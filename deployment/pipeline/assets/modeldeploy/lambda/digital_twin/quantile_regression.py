import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class QuantileRegression(BaseEstimator, RegressorMixin):
    """Linear regression model that predicts conditional quantiles.
    The linear :class:`QuantileRegression` optimizes the pinball loss for a
    desired `quantile` and is robust to outliers.
    Parameters
    ----------
    quantile : float, default=0.5
        The quantile that the model tries to predict. It must be strictly
        between 0 and 1. If 0.5 (default), the model predicts the 50%
        quantile, i.e. the median.
    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the features.
    intercept_: float value
        Estimated intercept.
    Examples
    --------
    >>> import numpy as np
    >>> n_samples, n_features = 10, 2
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> reg = QuantileRegression(quantile=0.8).fit(X, y)
    >>> np.mean(y <= reg.predict(X))
    0.8
    """

    def __init__(self, quantile=0.5):
        """Initialize the class.

        Args:
            :quantile:          the quantile of the quantile forrest, default: 0.5

        """
        self.quantile = quantile

    def quantile_loss(self, coefs, X, y):
        """Define the quantile loss function for optimization
        Parameters
        ----------
        X     : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data.
        y     : array-like of shape (n_samples,)
                Target values.
        coefs : array-like of shape (n_features+1,)
                Regression coefficients.
        Returns
        -------
        mean : array-like of shape (n_features+1,)
            Returns the calculated mean.
        """

        preds = X @ coefs[:-1] + coefs[-1]
        mean = np.nanmean(
            (preds >= y) * (1 - self.quantile) * (preds - y)
            + (preds < y) * self.quantile * (y - preds)
        )

        return mean

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Returns self.
        """
        # scikit-learn check X and y
        X, y = check_X_y(X, y)

        d = X.shape[1]
        x0 = np.repeat(0.0, d + 1)  # starting vector

        # Run optimization
        *self.coef_, self.intercept_ = minimize(
            self.quantile_loss, x0=x0, args=(X, y)
        ).x  # the heavy lifting

        return self

    def predict(self, X):
        """Predict with the trained model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : object
            Returns self.
        """
        # scikit-learn check if model is fitted
        check_is_fitted(self)

        # scikit-learn check X
        X = check_array(X)

        # Return X in |R^(nxn) times self.coef_ in |R^n + self.intercept_
        return X @ self.coef_ + self.intercept_
