# Changelog

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