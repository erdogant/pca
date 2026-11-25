import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from pca import pca


@pytest.fixture(scope="module")
def iris():
    d = load_iris()
    return pd.DataFrame(d.data, columns=[c.replace(" (cm)", "") for c in d.feature_names]), d.target_names[d.target]


def fitted(X, n_components=2, **kw):
    model = pca(n_components=n_components, verbose="silent", **kw)
    model.fit_transform(X)
    return model


def cosine(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


def plotted_points(ax, n):
    # scatterd draws one collection per class
    pts = np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections if len(c.get_offsets()) > 0])
    assert len(pts) == n
    return pts


def arrows(model, PC, scale):
    """Loadings as they are drawn in the biplot, before the common fit-to-range factor."""
    return model.results["loadings"].iloc[PC, :].mul(model._biplot_lam(PC, scale), axis=0)


def test_lam_follows_r_definition(iris):
    X, _ = iris
    model = fitted(X, 4)
    sdev = np.sqrt(model.results["model"].explained_variance_)
    n = len(X)
    np.testing.assert_allclose(model._biplot_lam([0, 1], 1), sdev[:2] * np.sqrt(n))
    np.testing.assert_allclose(model._biplot_lam([0, 1], 0.5), (sdev[:2] * np.sqrt(n)) ** 0.5)
    np.testing.assert_allclose(model._biplot_lam([0, 1], 0), [1, 1])
    np.testing.assert_allclose(model._biplot_lam([1, 3], 1), sdev[[1, 3]] * np.sqrt(n))  # PC selection
    # prcomp(iris[, 1:4]) reports sdev = 2.0563, 0.4926, 0.2797, 0.1544
    np.testing.assert_allclose(sdev, [2.0563, 0.4926, 0.2797, 0.1544], atol=1e-4)


@pytest.mark.parametrize("scale", [-0.1, 1.5, "a", [1]])
def test_invalid_scale_raises(iris, scale):
    X, _ = iris
    model = fitted(X)
    with pytest.raises(ValueError, match="scale"):
        model.biplot(scale=scale, visible=False)
    plt.close("all")


def test_arrows_relate_to_covariance(iris):
    X, _ = iris
    model = fitted(X, 4)
    A = arrows(model, [0, 1, 2, 3], 1).values  # PCs x features
    # With all PCs, the inner products of the arrows are exactly n * covariance of the features.
    np.testing.assert_allclose(A.T @ A / len(X), np.cov(X.values, rowvar=False), atol=1e-10)


def test_issue_example_sepal_length_petal_length(iris):
    X, _ = iris
    model = fitted(X, 2)
    corr = X.corr().loc["sepal length", "petal length"]  # 0.87
    raw = model.results["loadings"]
    cos_raw = cosine(raw["sepal length"].values, raw["petal length"].values)
    A = arrows(model, [0, 1], 1)
    cos_scaled = cosine(A["sepal length"].values, A["petal length"].values)
    assert cos_raw < 0.5  # almost orthogonal without the scaling
    assert cos_scaled == pytest.approx(corr, abs=0.05)
    # Arrow length reflects the standard deviation of the feature.
    lengths = np.linalg.norm(A.values, axis=0)
    assert lengths.argmax() == list(X.columns).index("petal length")


def test_normalized_data_arrows_approximate_correlation(iris):
    X, _ = iris
    model = fitted(X, 4, normalize=True)
    A = arrows(model, [0, 1, 2, 3], 1).values
    # StandardScaler normalizes with n and the PCA variances use n - 1, hence the factor n / (n - 1).
    n = len(X)
    np.testing.assert_allclose(A.T @ A / n, np.corrcoef(X.values, rowvar=False) * n / (n - 1), atol=1e-10)


def test_samples_are_scaled_and_arrows_fit_their_range(iris):
    X, y = iris
    model = fitted(X, 3)
    for scale in (0, 0.5, 1):
        fig, ax = model.biplot(labels=y, scale=scale, visible=True)
        lam = model._biplot_lam([0, 1], scale)
        sc = model.results["PC"].iloc[:, [0, 1]].values / lam
        pts = plotted_points(ax, len(X))
        np.testing.assert_allclose(np.sort(pts[:, 0]), np.sort(sc[:, 0]), atol=1e-10)
        np.testing.assert_allclose(np.sort(pts[:, 1]), np.sort(sc[:, 1]), atol=1e-10)
        # Arrow vertices stay within the range of the samples (up to the arrow head width) and reach it in one dimension.
        verts = np.concatenate([p.get_xy() for p in ax.patches])
        for j in (0, 1):
            limit = np.abs(sc[:, j]).max()
            assert np.abs(verts[:, j]).max() <= limit * 1.06
        assert max(np.abs(verts[:, j]).max() / np.abs(sc[:, j]).max() for j in (0, 1)) > 0.95
        plt.close("all")


def test_default_is_unchanged(iris):
    X, y = iris
    model = fitted(X, 3)
    fig, ax = model.biplot(labels=y)
    sc = model.results["PC"].iloc[:, [0, 1]].values
    pts = plotted_points(ax, len(X))
    np.testing.assert_allclose(np.sort(pts[:, 0]), np.sort(sc[:, 0]), atol=1e-10)
    plt.close("all")


def test_arrowdict_scale_factor_is_respected(iris):
    X, y = iris
    model = fitted(X, 2)
    extents = []
    for sf in (1.0, 0.5):
        fig, ax = model.biplot(labels=y, scale=1, arrowdict={"scale_factor": sf})
        verts = np.concatenate([p.get_xy() for p in ax.patches])
        extents.append(np.abs(verts[:, 0]).max())
        plt.close("all")
    assert extents[1] == pytest.approx(extents[0] * 0.5, rel=0.1)


def test_scatter_scale_and_3d_and_pc_selection(iris):
    X, y = iris
    model = fitted(X, 4)
    fig, ax = model.scatter(labels=y, scale=1)
    fig, ax = model.biplot3d(labels=y, scale=1)
    fig, ax = model.biplot(labels=y, scale=0.5, PC=[1, 2])
    plt.close("all")


@pytest.mark.parametrize("method", ["trunc_svd", "sparse_pca"])
def test_other_methods(iris, method):
    X, y = iris
    model = fitted(X, 2, method=method)
    fig, ax = model.biplot(labels=y, scale=1)
    plt.close("all")


def test_logistic_pca(iris):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 2))
    X = (rng.random((200, 12)) < 1 / (1 + np.exp(-(Z @ rng.normal(size=(2, 12)))))).astype(float)
    model = fitted(X, 2, method="logistic_pca")
    # No explained variance available: sdev is taken from the scores.
    lam = model._biplot_lam([0, 1], 1)
    np.testing.assert_allclose(lam, model.results["PC"].std(ddof=1).values * np.sqrt(200))
    fig, ax = model.biplot(scale=1)
    plt.close("all")


def test_scores_of_transformed_samples_are_scaled_like_fitted_samples(iris):
    X, y = iris
    model = fitted(X, 2)
    lam_before = model._biplot_lam([0, 1], 1)
    model.transform(X.iloc[:10], update_outlier_params=False)
    # The scaling factors for PCA come from the fitted model, but n counts all plotted samples.
    np.testing.assert_allclose(model._biplot_lam([0, 1], 1) / lam_before, np.sqrt(160 / 150))
    fig, ax = model.biplot(scale=1)
    plt.close("all")


def drawn(ax):
    """Everything that is drawn: the points and the arrow polygons."""
    pts = np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections if len(c.get_offsets()) > 0])
    verts = np.concatenate([p.get_xy() for p in ax.patches]) if ax.patches else np.zeros((0, 2))
    return pts, verts


def test_boolean_scale_true_is_one_and_false_is_none(iris):
    X, y = iris
    model = fitted(X, 2)
    ref = {}
    for key, scale in (("one", 1), ("none", None)):
        fig, ax = model.biplot(labels=y, scale=scale)
        ref[key] = drawn(ax)
        plt.close("all")
    for key, scale in (("one", True), ("none", False)):
        fig, ax = model.biplot(labels=y, scale=scale)
        pts, verts = drawn(ax)
        np.testing.assert_allclose(pts, ref[key][0])
        np.testing.assert_allclose(verts, ref[key][1])
        plt.close("all")
    # The two modes really differ.
    assert not np.allclose(ref["one"][0], ref["none"][0])
    fig, ax = model.scatter(labels=y, scale=True)
    fig, ax = model.scatter(labels=y, scale=False)
    plt.close("all")


def test_reported_usage_with_outliers_and_scale_true(iris):
    X, y = iris
    model = pca(n_components=2, verbose="silent", n_std=2)
    model.fit_transform(X)
    fig, ax = model.biplot(s=200, SPE=True, HT2=True, scale=True)
    plt.close("all")
