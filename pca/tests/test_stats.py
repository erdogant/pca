
import numpy as np
import pandas as pd
import pca as pca_module
from pca import pca
import importlib
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_iris
pca_module = importlib.import_module("pca.pca")

def _get_test_data():
    """Create deterministic data with correlated features."""
    rng = np.random.default_rng(42)

    x1 = rng.normal(size=200)
    x2 = 0.8 * x1 + rng.normal(scale=0.3, size=200)
    x3 = -0.5 * x1 + rng.normal(scale=0.5, size=200)
    x4 = rng.normal(size=200)
    x5 = 0.5 * x2 + 0.3 * x4 + rng.normal(scale=0.2, size=200)

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
    })


def test_standard_pca_statistics():
    """Check explained variance statistics for standard PCA."""
    X = _get_test_data()

    model = pca(
        n_components=3,
        method="pca",
        detect_outliers=None,
        random_state=42,
        verbose="silent",
    )

    results = model.fit_transform(X)

    explained_var = np.asarray(results["explained_var"])
    variance_ratio = np.asarray(results["variance_ratio"])
    model_ratio = np.asarray(results["model"].explained_variance_ratio_)

    assert explained_var.shape == (3,)
    assert variance_ratio.shape == (3,)
    assert model_ratio.shape == (3,)

    assert np.all(np.isfinite(explained_var))
    assert np.all(np.isfinite(variance_ratio))
    assert np.all(np.isfinite(model_ratio))

    # Explained variance ratios must sum to one across all fitted PCs.
    # Explained variance ratios must not exceed the total variance.
    assert model_ratio.sum() <= 1.0 + 1e-8
    assert model_ratio.sum() > 0


    # Cumulative explained variance must equal the cumulative sum.
    np.testing.assert_allclose(
        explained_var,
        np.cumsum(model_ratio),
        atol=1e-8,
    )

    # Individual explained variance must equal the difference
    # between consecutive cumulative values.
    np.testing.assert_allclose(
        variance_ratio,
        np.diff(explained_var, prepend=0),
        atol=1e-8,
    )

    # pcp must represent the cumulative variance of the selected PCs.
    np.testing.assert_allclose(
        results["pcp"],
        explained_var[-1],
        atol=1e-8,
    )


def test_sparse_pca_adjusted_statistics():
    """Check adjusted explained variance statistics for Sparse PCA."""
    X = _get_test_data()

    model = pca(
        n_components=3,
        method="sparse_pca",
        detect_outliers=None,
        random_state=42,
        verbose="silent",
    )

    results = model.fit_transform(X)

    explained_var = np.asarray(results["explained_var"])
    variance_ratio = np.asarray(results["variance_ratio"])
    model_ratio = np.asarray(results["model"].explained_variance_ratio_)

    assert explained_var.shape == (3,)
    assert variance_ratio.shape == (3,)
    assert model_ratio.shape == (3,)

    # Adjusted statistics must be available.
    assert np.all(np.isfinite(explained_var))
    assert np.all(np.isfinite(variance_ratio))
    assert np.all(np.isfinite(model_ratio))

    # Ratios must be non-negative and normalized.
    assert np.all(model_ratio >= 0)
    np.testing.assert_allclose(model_ratio.sum(), 1.0, atol=1e-8)

    # Cumulative and individual statistics must be consistent.
    np.testing.assert_allclose(
        explained_var,
        np.cumsum(model_ratio),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        variance_ratio,
        np.diff(explained_var, prepend=0),
        atol=1e-8,
    )

    # Sparse PCA must retain its adjusted cumulative variance.
    assert results["pcp"] is not None
    np.testing.assert_allclose(
        results["pcp"],
        explained_var[-1],
        atol=1e-8,
    )

    # The adjusted statistics should not be replaced by dummy values.
    assert np.any(model_ratio > 0)



def test_sparse_pca_statistics_fallback(monkeypatch):
    """Check fallback behavior when adjusted variance cannot be computed."""
    import importlib

    # Import the actual module containing _get_explained_variance.
    pca_module = importlib.import_module("pca.pca")

    def raise_error(*args, **kwargs):
        raise ValueError("Test error")

    monkeypatch.setattr(
        pca_module,
        "_get_explained_variance",
        raise_error,
    )

    X = _get_test_data()

    model = pca_module.pca(
        n_components=3,
        method="sparse_pca",
        detect_outliers=None,
        random_state=42,
        verbose="silent",
    )

    results = model.fit_transform(X)

    # Failed adjusted statistics must be represented as unavailable.
    assert results["explained_var"] is None
    assert results["variance_ratio"] is None
    assert results["pcp"] is None

    # The model should still provide a correctly sized fallback array.
    assert results["model"].explained_variance_ratio_.shape == (3,)

    np.testing.assert_array_equal(
        results["model"].explained_variance_ratio_,
        np.zeros(3),
    )


@pytest.fixture
def iris_data():
    """Load a deterministic dataset for PCA tests."""
    return load_iris().data


def test_standard_pca_single_component(iris_data):
    """Test explained variance statistics with a single component."""
    model = pca_module.pca(
        n_components=1,
        detect_outliers=None,
        verbose="silent",
    )

    results = model.fit_transform(iris_data)

    assert results["PC"].shape[1] == 1
    assert len(results["explained_var"]) == 1
    assert len(results["variance_ratio"]) == 1

    assert np.isfinite(results["explained_var"]).all()
    assert np.isfinite(results["variance_ratio"]).all()

    assert results["explained_var"][0] == pytest.approx(
        results["variance_ratio"][0]
    )

    assert 0 < results["explained_var"][0] < 1


def test_sparse_pca_zero_adjusted_variance(iris_data, monkeypatch):
    """Test Sparse PCA when adjusted explained variance is zero."""

    def return_zero_variance(X, components):
        return np.zeros(components.shape[0])

    monkeypatch.setattr(
        pca_module,
        "_get_explained_variance",
        return_zero_variance,
    )

    model = pca_module.pca(
        n_components=3,
        method="sparse_pca",
        detect_outliers=None,
        random_state=42,
        verbose="silent",
    )

    results = model.fit_transform(iris_data)

    assert results["explained_var"] is None
    assert results["variance_ratio"] is None
    assert results["pcp"] is None

    assert np.array_equal(
        results["model"].explained_variance_ratio_,
        np.zeros(3),
    )


def test_standard_pca_component_selection(iris_data):
    """Test automatic component selection based on explained variance."""
    threshold = 0.80

    model = pca_module.pca(
        n_components=threshold,
        detect_outliers=None,
        verbose="silent",
    )

    results = model.fit_transform(iris_data)

    explained_var = results["explained_var"]

    assert results["PC"].shape[1] == model.n_components
    assert explained_var[model.n_components - 1] >= threshold

    if model.n_components > 1:
        assert explained_var[model.n_components - 2] < threshold


def test_standard_pca_plot(iris_data):
    """Test that the explained variance plot is generated."""
    model = pca_module.pca(
        n_components=3,
        detect_outliers=None,
        verbose="silent",
    )

    model.fit_transform(iris_data)

    try:
        fig, ax = model.plot(visible=False)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0
        assert len(ax.patches) > 0
    finally:
        plt.close("all")


def test_sparse_pca_plot_without_adjusted_variance(iris_data, monkeypatch):
    """Test that plotting is skipped when adjusted variance is unavailable."""

    def raise_error(X, components):
        raise ValueError("Test error")

    monkeypatch.setattr(
        pca_module,
        "_get_explained_variance",
        raise_error,
    )

    model = pca_module.pca(
        n_components=3,
        method="sparse_pca",
        detect_outliers=None,
        random_state=42,
        verbose="silent",
    )

    model.fit_transform(iris_data)

    fig, ax = model.plot(visible=False)

    assert fig is None
    assert ax is None

    plt.close("all")
