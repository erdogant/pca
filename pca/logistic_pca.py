"""Logistic PCA for binary data with native support for missing values.

Implements Landgraf & Lee (2015), "Dimensionality Reduction for Binary Data
through the Projection of Natural Parameters" (https://arxiv.org/abs/1510.06112).
"""

import numpy as np
from scipy.special import expit, xlogy


def _to_binary_array(X):
    """Convert input to a float array with values in {0, 1, nan}.

    Accepts numpy arrays, lists and DataFrames holding 0/1, bool and missing values
    (np.nan, None, pd.NA). Raises ValueError for anything else.
    """
    import pandas as pd

    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"[pca] >Error: Logistic PCA requires a 2D input, got {arr.ndim} dimension(s).")
    try:
        if arr.dtype == object:
            missing = pd.isna(arr)
            vals = np.where(missing, np.nan, arr).astype(float)
        else:
            vals = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError("[pca] >Error: Logistic PCA requires binary data: only 0, 1, True, False or missing values are allowed.")

    invalid = ~np.isnan(vals) & (vals != 0) & (vals != 1)
    if invalid.any():
        raise ValueError(
            f"[pca] >Error: Logistic PCA requires binary data. Found {invalid.sum()} value(s) that are not 0, 1 or missing, "
            f"for example: {np.unique(vals[invalid])[:5].tolist()}."
        )
    return vals


def _bernoulli_deviance(Xo, Theta, obs):
    """Bernoulli deviance over observed entries only. Uses log(1+exp(t)) = logaddexp(0, t) for stability."""
    return float(np.sum(obs * (-2.0 * Xo * Theta + 2.0 * np.logaddexp(0.0, Theta))))


def _null_deviance(X):
    """Deviance of the intercept-only model (column-wise observed proportion) over observed entries only."""
    obs = ~np.isnan(X)
    n_obs = obs.sum(axis=0)
    p = np.where(obs, X, 0.0).sum(axis=0) / n_obs
    # xlogy handles 0*log(0)=0 for constant columns.
    return float(np.sum(-2.0 * n_obs * (xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))))


class LogisticPCA:
    """Logistic PCA (Landgraf & Lee, 2015) for binary data with missing values.

    The natural parameters are modelled as ``Theta = 1 mu^T + (Theta_tilde - 1 mu^T) U U^T`` with
    ``Theta_tilde = m * (2X - 1)``, and estimated by minimizing the Bernoulli deviance over the
    observed entries with a majorization-minimization algorithm. Missing entries do not contribute
    to the deviance. During fitting, the working natural parameter of a missing entry is set to its
    current fitted value (the data are never imputed).

    Parameters
    ----------
    n_components : int, default=2
        Number of components.
    m : float, default=4
        Approximation of the saturated natural parameters. Larger values approximate the saturated model better.
    max_iter : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-6
        Relative change in deviance below which the algorithm stops.

    Attributes
    ----------
    mu_ : ndarray (n_features,)
        Column offsets (natural parameter scale).
    components_ : ndarray (n_components, n_features)
        Orthonormal loadings.
    scores_ : ndarray (n_samples, n_components)
        Scores of the training data.
    fitted_ : ndarray (n_samples, n_features)
        Fitted probabilities of the training data, including missing cells.
    deviance_, null_deviance_, deviance_explained_ : float
        Model deviance, intercept-only deviance and ``1 - deviance_/null_deviance_`` (observed entries only).
    n_iter_, converged_ : int, bool
    X_ : ndarray
        Training data as float array with nan for missing.
    """

    def __init__(self, n_components=2, m=4, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """Fit the model."""
        X = _to_binary_array(X)
        n, d = X.shape
        k = self.n_components
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"[pca] >Error: Logistic PCA requires n_components to be an integer >= 1, got {k!r}.")
        if k > min(n, d):
            raise ValueError(f"[pca] >Error: n_components={k} cannot exceed min(n_samples, n_features)={min(n, d)}.")
        if not np.isscalar(self.m) or not np.isfinite(self.m) or self.m <= 0:
            raise ValueError(f"[pca] >Error: m must be a positive number, got {self.m!r}.")

        obs = ~np.isnan(X)
        self._check_not_empty(obs)
        Xo = np.where(obs, X, 0.0)
        theta_obs = self.m * (2.0 * Xo - 1.0)

        # Initialization: missing entries start at the column offset, U from the SVD of the centered Theta_tilde.
        mu = (theta_obs * obs).sum(axis=0) / obs.sum(axis=0)
        T = np.where(obs, theta_obs, mu)
        U = np.linalg.svd(T - mu, full_matrices=False)[2][:k].T

        prev = np.inf
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            Theta = mu + ((T - mu) @ U) @ U.T
            # Working response of the majorization (bound 1/4 on the logistic variance); zero update for missing cells.
            Z = Theta + 4.0 * (Xo - expit(Theta)) * obs
            mu = (Z - (T @ U) @ U.T).mean(axis=0)
            Tc, Zc = T - mu, Z - mu
            M = Tc.T @ Zc
            M = M + M.T - Tc.T @ Tc
            U = np.linalg.eigh(M)[1][:, ::-1][:, :k]

            Theta = mu + (Tc @ U) @ U.T
            T = np.where(obs, theta_obs, Theta)
            dev = _bernoulli_deviance(Xo, Theta, obs)
            if np.isfinite(prev) and abs(prev - dev) <= self.tol * max(abs(prev), 1.0):
                self.converged_ = True
                break
            prev = dev
        self.n_iter_ = it

        self.mu_ = mu
        self.components_ = U.T
        self.X_ = X
        self.scores_ = self._scores(X)
        Theta = self.mu_ + self.scores_ @ self.components_
        self.fitted_ = expit(Theta)
        self.deviance_ = _bernoulli_deviance(Xo, Theta, obs)
        self.null_deviance_ = _null_deviance(X)
        self.deviance_explained_ = 1.0 - self.deviance_ / self.null_deviance_ if self.null_deviance_ > 0 else np.nan
        return self

    def transform(self, X):
        """Component scores for new observations. Rows may contain missing values but not be entirely missing."""
        X = _to_binary_array(X)
        if X.shape[1] != self.mu_.shape[0]:
            raise ValueError(f"[pca] >Error: X has {X.shape[1]} features but the model was fitted with {self.mu_.shape[0]}.")
        self._check_not_empty(~np.isnan(X), check_columns=False)
        return self._scores(X)

    def predict_proba(self, X):
        """Estimated probabilities for all cells of X, including missing cells."""
        return expit(self.mu_ + self.transform(X) @ self.components_)

    def _scores(self, X):
        # Scores s_i = argmin ||(Theta_tilde_i - mu)_obs - U_obs s||^2 over observed features only.
        # For complete rows this equals the standard projection (Theta_tilde_i - mu) U.
        obs = ~np.isnan(X)
        U = self.components_.T
        R = self.m * (2.0 * np.where(obs, X, 0.0) - 1.0) - self.mu_
        if obs.all():
            return R @ U
        S = np.empty((X.shape[0], U.shape[1]))
        patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        for p, pattern in enumerate(patterns):
            rows = inverse == p
            S[rows] = R[rows][:, pattern] @ np.linalg.pinv(U[pattern, :]).T
        return S

    @staticmethod
    def _check_not_empty(obs, check_columns=True):
        empty_rows = np.where(~obs.any(axis=1))[0]
        if len(empty_rows) > 0:
            raise ValueError(f"[pca] >Error: Rows without any observed value are not supported (row index: {empty_rows[:10].tolist()}).")
        if check_columns:
            empty_cols = np.where(~obs.any(axis=0))[0]
            if len(empty_cols) > 0:
                raise ValueError(f"[pca] >Error: Columns without any observed value are not supported (column index: {empty_cols[:10].tolist()}).")
