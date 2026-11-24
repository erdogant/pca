import logging
from pca.pca import pca

from pca.pca import (
    import_example,
    hotellingsT2,
    spe_dmodx,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.11.0'

# Setup root logger
_logger = logging.getLogger('pca')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False


# module level doc-string
__doc__ = """
pca
=====================================================================

pca: A Python Package for Principal Component Analysis.

Examples
--------
>>> from pca import pca
>>>
>>> Initialize
>>> model = pca(n_components=3)
>>>
>>> # Load example data
>>> df = model.import_example(data='iris')
>>>
>>> # Fit using PCA
>>> results = model.fit_transform(df)
>>>
>>> # Scree plot together with explained variance.
>>> fig, ax = model.plot()
>>>
>>> # Plot loadings
>>> marker=df['label']
>>> labels=df['label']
>>>
>>> fig, ax = model.biplot()
>>> fig, ax = model.biplot(density=True, SPE=True, HT2=True, marker=marker, labels=labels)
>>> fig, ax = model.scatter(marker=marker, labels=labels)
>>>
>>> # 3D plots
>>> fig, ax = model.scatter3d(marker=marker, labels=labels)
>>> fig, ax = model.biplot3d()
>>>
>>> # Logistic PCA for binary data (missing values are supported natively)
>>> model = pca(n_components=2, method='logistic_pca', m=4)
>>> results = model.fit_transform(X_binary)  # X_binary contains 0, 1 and np.nan
>>> results['deviance_explained'], results['fitted']
>>> fig, ax = model.plot()  # Deviance explained versus number of components
>>> fig, ax = model.biplot()
>>>

References
----------
* Blog: erdogant.medium.com
* Github: https://github.com/erdogant/pca
* Documentation: https://erdogant.github.io/pca/

"""
