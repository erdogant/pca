# Changelog

## [2.11.0] 26-11-2026

### New
- Added Logistic PCA (`method='logistic_pca'`) for binary data with native missing-value support.
- Added deviance-based component selection for Logistic PCA when `n_components` is a fraction (0-1).
- Added probability outputs for Logistic PCA (`fitted`) and deviance metrics (`deviance_explained`, `deviance_explained_path`).

### Improvements
- Added `scale` support to `biplot`, `scatter`, and 3D biplot behavior for improved control of scores/loadings visualization.
- Improved biplot scaling behavior and consistency across PCA methods.
- Updated examples and API documentation for Logistic PCA and scaled plotting workflows.

### Fixes
- Fixed transform/outlier behavior so existing fitted outlier parameters can be kept stable with `update_outlier_params=False`.
- Improved robustness and validation for Logistic PCA inputs and parameter handling.

### Tests
- Added comprehensive Logistic PCA test coverage (input validation, missing values, transforms, outputs, plotting).
- Added dedicated tests for biplot scaling behavior and `scale` semantics.
- Added regression tests for transform + outlier parameter update behavior.

## [2.10.3] 23-11-2026

### Improvements
- Improved explained variance calculation and result handling.
- Added adjusted explained variance calculation for SparsePCA.
- Improved handling of zero or unavailable adjusted explained variance.
- Prevented misleading explained variance results when the adjusted variance calculation fails.
- Improved plotting behavior when explained variance is unavailable.

### Tests
- Added tests for standard PCA explained variance statistics.
- Added tests for SparsePCA adjusted explained variance.
- Added coverage for zero variance and calculation failures.
- Added tests for automatic component selection based on explained variance.
- Added tests for explained variance plotting, including SparsePCA fallback behavior.