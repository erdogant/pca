import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as _assert_frame_equal

from pca import pca


def assert_frame_equal(left, right):
    # New samples are labeled 'mapped', which turns an integer index into an object index; the values must be identical.
    _assert_frame_equal(left, right, check_index_type=False)


def make_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randint(low=1, high=10, size=(10000, 10)))
    X.iloc[500:510, 8:] = 15  # outliers
    model = pca(n_components=5, alpha=0.05, n_std=3, normalize=True, random_state=42, verbose="silent")
    model.fit_transform(X)
    X_new = pd.DataFrame(np.random.randint(low=1, high=10, size=(1000, 10)))
    return model, X, X_new


def test_locked_params_keep_outliers_of_fitted_data():
    model, X, X_new = make_model()
    before = model.results["outliers"].copy()
    n = len(X)
    model.transform(X_new, update_outlier_params=False)
    after = model.results["outliers"]
    assert len(after) == n + len(X_new)
    assert_frame_equal(before, after.iloc[:n])  # all columns, incl. y_proba and y_bool
    assert before["y_bool"].sum() == after.iloc[:n]["y_bool"].sum() == 1


def test_locked_params_repeated_transforms_keep_all_previous_results():
    model, X, X_new = make_model()
    model.transform(X_new, update_outlier_params=False)
    snapshot = model.results["outliers"].copy()
    model.transform(X_new.iloc[:200], update_outlier_params=False)
    assert_frame_equal(snapshot, model.results["outliers"].iloc[: len(snapshot)])
    assert len(model.results["outliers"]) == len(snapshot) + 200
    assert len(model.results["PC"]) == len(model.results["outliers"])


def test_locked_params_are_not_changed_and_new_outliers_are_detected():
    model, X, X_new = make_model()
    params = {k: v for k, v in model.results["outliers_params"].items()}
    X_out = X_new.copy()
    X_out.iloc[0, :] = 15
    model.transform(X_out, update_outlier_params=False)
    assert model.results["outliers_params"]["paramT2"] == params["paramT2"]
    assert model.results["outliers"].iloc[len(X)]["y_bool_spe"]  # extreme new sample flagged


def test_update_params_still_updates_and_stores_params():
    model, X, X_new = make_model()
    old = model.results["outliers_params"]["paramT2"]
    model.transform(X_new, update_outlier_params=True)
    new = model.results["outliers_params"]
    assert new["paramT2"] is not None and new["paramSPE"] is not None
    assert new["paramT2"] != old
    assert len(model.results["outliers"]) == len(X) + len(X_new)
    # A later locked call keeps the updated parameters and the outliers of all previous samples.
    snapshot = model.results["outliers"].copy()
    model.transform(X_new.iloc[:100], update_outlier_params=False)
    assert model.results["outliers_params"]["paramT2"] == new["paramT2"]
    assert_frame_equal(snapshot, model.results["outliers"].iloc[: len(snapshot)])


def test_locked_transform_with_single_outlier_method():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(300, 6))
    for method in (["ht2"], ["spe"]):
        model = pca(n_components=3, detect_outliers=method, verbose="silent")
        model.fit_transform(X)
        before = model.results["outliers"].copy()
        model.transform(pd.DataFrame(np.random.randn(50, 6)), update_outlier_params=False)
        assert_frame_equal(before, model.results["outliers"].iloc[:300])
        assert len(model.results["outliers"]) == 350


def test_locked_transform_logistic_pca():
    rng = np.random.default_rng(0)
    X = (rng.random((200, 10)) < 0.4).astype(float)
    model = pca(n_components=2, method="logistic_pca", verbose="silent")
    model.fit_transform(X)
    before = model.results["outliers"].copy()
    model.transform((rng.random((30, 10)) < 0.4).astype(float), update_outlier_params=False)
    assert_frame_equal(before, model.results["outliers"].iloc[:200])
