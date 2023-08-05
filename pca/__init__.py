from pca.pca import pca

from pca.pca import (
    import_example,
    hotellingsT2,
    spe_dmodx,
    )


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.0.4'

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
>>> fig, ax = model.biplot()
>>> fig, ax = model.biplot(density=True, SPE=True, HT2=True)
>>> fig, ax = model.scatter()
>>>
>>> 3D plots
>>> fig, ax = model.scatter3d()
>>> fig, ax = model.biplot3d(density=True, SPE=True, HT2=True)
>>>
>>> # Normalize out PCs
>>> X_norm = model.norm(X)

References
----------
* Blog: https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559
* Github: https://github.com/erdogant/pca
* Documentation: https://erdogant.github.io/pca/

"""
