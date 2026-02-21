import logging
from pca.pca import pca

from pca.pca import (
    import_example,
    hotellingsT2,
    spe_dmodx,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.10.2'

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

References
----------
* Blog: erdogant.medium.com
* Github: https://github.com/erdogant/pca
* Documentation: https://erdogant.github.io/pca/

"""
