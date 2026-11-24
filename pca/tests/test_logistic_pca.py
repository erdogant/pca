import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from pca import pca
from pca.logistic_pca import LogisticPCA


def make_data(n=300, d=15, k=2, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, k))
    W = rng.normal(size=(k, d)) * 1.5
    mu = rng.normal(size=d) * 0.5
    P = expit(mu + Z @ W)
    X = (rng.random((n, d)) < P).astype(float)
    return X, Z, P


def fit(X, n_components=2, **kw):
    model = pca(n_components=n_components, method="logistic_pca", verbose="silent", **kw)
    return model, model.fit_transform(X)


@pytest.fixture(scope="module")
def data():
    return make_data()


# --- Input types -----------------------------------------------------------------------------------------------
def test_numpy_binary(data):
    X, _, _ = data
    _, out = fit(X)
    assert out["PC"].shape == (X.shape[0], 2)
    assert out["loadings"].shape == (2, X.shape[1])


def test_boolean_input(data):
    X, _, _ = data
    _, out_f = fit(X)
    _, out_b = fit(X.astype(bool))
    np.testing.assert_allclose(out_f["PC"].values, out_b["PC"].values, atol=1e-8)


def test_dataframe_labels(data):
    X, _, _ = data
    df = pd.DataFrame(X.astype(int), columns=[f"f{i}" for i in range(X.shape[1])], index=[f"s{i}" for i in range(X.shape[0])])
    _, out = fit(df)
    assert list(out["loadings"].columns) == list(df.columns)
    assert list(out["PC"].index) == list(df.index)
    assert list(out["fitted"].columns) == list(df.columns)


def test_nan_none_and_pandas_na(data):
    X, _, _ = data
    Xo = X.astype(object)
    Xo[0, 0] = None
    Xo[1, 1] = np.nan
    Xo[2, 2] = pd.NA
    _, out = fit(Xo)
    assert np.isfinite(out["PC"].values).all()
    df = pd.DataFrame(X).astype("Int64")
    df.iloc[3, 3] = pd.NA
    _, out = fit(df)
    assert np.isfinite(out["PC"].values).all()


def test_invalid_values_raise(data):
    X, _, _ = data
    Xb = X.copy()
    Xb[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        fit(Xb)
    Xb[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        fit(Xb)
    Xb[0, 0] = np.inf
    with pytest.raises(ValueError, match="binary"):
        fit(Xb)
    with pytest.raises(ValueError, match="binary"):
        fit(pd.DataFrame(X).astype(object).replace({1.0: "a"}))


# --- Parameter validation ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_components", [0.95, None, 0, 2.0, True, 100])
def test_n_components_must_be_valid_integer(data, n_components):
    X, _, _ = data
    with pytest.raises(ValueError):
        fit(X, n_components=n_components)


def test_unsupported_options(data):
    X, _, _ = data
    with pytest.raises(ValueError):
        fit(X, normalize=True)
    with pytest.raises(ValueError):
        fit(X, m=-1)
    import scipy.sparse as sp

    with pytest.raises(ValueError):
        fit(sp.csr_matrix(X))


def test_onehot_does_not_override_method(data):
    X, _, _ = data
    model, _ = fit(X, onehot=True)
    assert model.method == "logistic_pca"


# --- Missing values ----------------------------------------------------------------------------------------------
def test_entirely_missing_row_and_column_raise(data):
    X, _, _ = data
    Xr = X.copy()
    Xr[5, :] = np.nan
    with pytest.raises(ValueError, match="Rows"):
        fit(Xr)
    Xc = X.copy()
    Xc[:, 3] = np.nan
    with pytest.raises(ValueError, match="Columns"):
        fit(Xc)


def test_missing_values_do_not_contribute_to_deviance(data):
    X, _, _ = data
    rng = np.random.default_rng(1)
    mask = rng.random(X.shape) < 0.15
    Xm = X.copy()
    Xm[mask] = np.nan
    model, out = fit(Xm)
    P = out["fitted"].values
    obs = ~mask
    manual = -2 * np.sum(X[obs] * np.log(P[obs]) + (1 - X[obs]) * np.log(1 - P[obs]))
    assert model.results["model"].deviance_ == pytest.approx(manual, rel=1e-8)
    # Null deviance on the same observed entries
    p = np.array([X[obs[:, j], j].mean() for j in range(X.shape[1])])
    n_obs = obs.sum(axis=0)
    null = -2 * np.sum(n_obs * (p * np.log(p) + (1 - p) * np.log(1 - p)))
    assert out["deviance_explained"] == pytest.approx(1 - manual / null, rel=1e-8)
    # Changing values in the missing cells (before masking) cannot change the result.
    X2 = X.copy()
    X2[mask] = 1 - X2[mask]
    X2[mask] = np.nan
    _, out2 = fit(X2)
    np.testing.assert_allclose(out["PC"].values, out2["PC"].values)


def test_fitted_probabilities_for_missing_cells(data):
    X, _, P = data
    rng = np.random.default_rng(2)
    mask = rng.random(X.shape) < 0.15
    Xm = X.copy()
    Xm[mask] = np.nan
    _, out = fit(Xm)
    fitted = out["fitted"].values
    assert np.isfinite(fitted[mask]).all()
    assert np.corrcoef(fitted[mask], P[mask])[0, 1] > 0.6


# --- Transform ---------------------------------------------------------------------------------------------------
def test_transform_complete_matches_fit(data):
    X, _, _ = data
    model, out = fit(X)
    PCs = model.transform(X, update_outlier_params=False)
    np.testing.assert_allclose(PCs.values, out["PC"].values[: X.shape[0]], atol=1e-8)


def test_transform_matches_projection_formula(data):
    X, _, _ = data
    model, out = fit(X)
    m = out["model"]
    expected = (4 * (2 * X - 1) - m.mu_) @ m.components_.T
    np.testing.assert_allclose(out["PC"].values, expected, atol=1e-8)


def test_transform_partially_missing(data):
    X, _, _ = data
    model, out = fit(X)
    new = X[:10].copy()
    new[:, :2] = np.nan
    new[0, 5] = np.nan
    PCs = model.transform(new)
    assert PCs.shape == (10, 2)
    assert np.isfinite(PCs.values).all()
    # Removing a few features should leave the scores correlated with the complete-row scores.
    full = out["PC"].values[:10]
    assert np.corrcoef(PCs.values[:, 0], full[:, 0])[0, 1] > 0.5
    # Probabilities for the missing cells of new observations are available and valid.
    P = out["model"].predict_proba(new)
    assert P.shape == new.shape and np.isfinite(P).all() and ((P >= 0) & (P <= 1)).all()


def test_transform_entirely_missing_row_rejected(data):
    X, _, _ = data
    model, _ = fit(X)
    new = X[:3].copy()
    new[1, :] = np.nan
    with pytest.raises(ValueError, match="Rows"):
        model.transform(new)


def test_transform_wrong_number_of_features(data):
    X, _, _ = data
    model, _ = fit(X)
    with pytest.raises(ValueError):
        model.transform(X[:, :-1])


# --- Results -----------------------------------------------------------------------------------------------------
def test_result_structure_and_probabilities(data):
    X, _, _ = data
    _, out = fit(X)
    for key in ["PC", "loadings", "model", "fitted", "deviance_explained", "explained_var", "variance_ratio", "topfeat", "outliers"]:
        assert key in out
    assert out["explained_var"] is None and out["variance_ratio"] is None
    assert ((out["fitted"].values >= 0) & (out["fitted"].values <= 1)).all()
    assert 0 < out["deviance_explained"] < 1
    # Orthonormal loadings
    L = out["loadings"].values
    np.testing.assert_allclose(L @ L.T, np.eye(2), atol=1e-8)


def test_deviance_calculation(data):
    X, _, _ = data
    _, out = fit(X)
    P = out["fitted"].values
    dev = -2 * np.sum(X * np.log(P) + (1 - X) * np.log(1 - P))
    assert out["model"].deviance_ == pytest.approx(dev, rel=1e-8)
    p = X.mean(axis=0)
    null = -2 * X.shape[0] * np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert out["deviance_explained"] == pytest.approx(1 - dev / null, rel=1e-8)


def test_deviance_explained_increases_with_components(data):
    X, _, _ = data
    de = [fit(X, n_components=k)[1]["deviance_explained"] for k in (1, 2, 3)]
    assert de[0] < de[1] < de[2]


def test_constant_column_and_constant_data():
    X, _, _ = make_data(n=100, d=6)
    X[:, 0] = 0
    _, out = fit(X)
    assert np.isfinite(out["deviance_explained"])
    _, out = fit(np.zeros((20, 4)), n_components=1)
    assert np.isnan(out["deviance_explained"])


def test_refit_overwrites_results(data):
    X, _, _ = data
    model, _ = fit(X)
    out = model.fit_transform(X[:100])
    assert out["PC"].shape[0] == 100


# --- Known latent structure --------------------------------------------------------------------------------------
def test_recovers_latent_structure():
    X, Z, P = make_data(n=600, d=30, k=2, seed=3)
    _, out = fit(X)
    S = np.c_[out["PC"].values, np.ones(len(X))]
    B = np.linalg.lstsq(S, Z, rcond=None)[0]
    r2 = 1 - ((Z - S @ B) ** 2).sum(0) / ((Z - Z.mean(0)) ** 2).sum(0)
    assert (r2 > 0.6).all()
    # Fitted probabilities approximate the true ones better than the column means do.
    err_model = np.mean((out["fitted"].values - P) ** 2)
    err_null = np.mean((X.mean(axis=0) - P) ** 2)
    assert err_model < 0.6 * err_null


# --- Numerical stability and convergence --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0.5, 4, 20, 100])
def test_numerical_stability(data, m):
    X, _, _ = data
    _, out = fit(X, m=m)
    assert np.isfinite(out["PC"].values).all()
    assert np.isfinite(out["fitted"].values).all()
    assert np.isfinite(out["deviance_explained"])


def test_perfectly_separable_data_is_finite():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, size=(200, 1))
    X = np.repeat(z, 8, axis=1).astype(float)
    _, out = fit(X, n_components=1)
    assert np.isfinite(out["PC"].values).all()
    assert out["deviance_explained"] > 0.9


def test_optimizer_converges_and_decreases_deviance(data):
    X, _, _ = data
    devs = []
    for it in (1, 5, 50):
        devs.append(LogisticPCA(2, max_iter=it, tol=0).fit(X).deviance_)
    assert devs[0] >= devs[1] >= devs[2]
    m = LogisticPCA(2).fit(X)
    assert m.converged_ and m.n_iter_ < 500
    Xm = X.copy()
    Xm[np.random.default_rng(0).random(X.shape) < 0.2] = np.nan
    assert LogisticPCA(2).fit(Xm).converged_


# --- Visualization -----------------------------------------------------------------------------------------------
def test_plots_run(data):
    X, _, _ = data
    model, out = fit(X, n_components=3)
    labels = np.where(out["PC"].values[:, 0] > 0, "a", "b")
    fig, ax = model.plot(visible=False)
    assert ax.get_ylabel() == "Proportion of deviance explained"
    assert len(out["deviance_explained_path"]) == 3
    assert out["deviance_explained_path"][-1] == pytest.approx(out["deviance_explained"])
    assert np.all(np.diff(out["deviance_explained_path"]) > 0)
    fig, ax = model.scatter(labels=labels, visible=False)
    assert "expl.var" not in ax.get_xlabel() and "deviance" in ax.get_title()
    fig, ax = model.biplot(labels=labels, visible=False)
    fig, ax = model.scatter3d(labels=labels, visible=False)
    fig, ax = model.biplot3d(labels=labels, visible=False)
    plt.close("all")


def test_norm_not_supported(data):
    X, _, _ = data
    model, _ = fit(X)
    with pytest.raises(NotImplementedError):
        model.norm(X)


# --- Regression tests for existing methods -------------------------------------------------------------------------
def test_regression_standard_pca():
    X = load_iris().data
    out = pca(n_components=2, verbose="silent").fit_transform(X)
    ref = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(np.abs(out["PC"].values), np.abs(ref.transform(X)), atol=1e-8)
    np.testing.assert_allclose(out["explained_var"], ref.explained_variance_ratio_.cumsum())
    assert "fitted" not in out and "deviance_explained" not in out


def test_regression_default_n_components_and_normalize():
    X = load_iris().data
    out = pca(verbose="silent", normalize=True).fit_transform(X)
    assert out["PC"].shape[1] == 2
    assert out["explained_var"][-1] >= 0.95


@pytest.mark.parametrize("method", ["trunc_svd", "sparse_pca"])
def test_regression_other_methods(method):
    X = load_iris().data
    out = pca(n_components=2, method=method, verbose="silent").fit_transform(X)
    assert out["PC"].shape == (150, 2)


def test_regression_onehot_still_selects_sparse_pca():
    model = pca(n_components=2, onehot=True, verbose="silent")
    assert model.method == "sparse_pca"


def test_regression_standard_plots_and_transform():
    X = load_iris().data
    model = pca(n_components=3, verbose="silent")
    model.fit_transform(X)
    fig, ax = model.plot(visible=False)
    fig, ax = model.biplot(visible=False)
    assert "expl.var" in ax.get_xlabel()
    assert model.transform(X[:5]).shape == (5, 3)
    plt.close("all")
