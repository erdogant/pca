from pca.pca import pca

from pca.pca import (
    import_example,
    hotellingsT2,
    spe_dmodx,
    )


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.9.0'

# module level doc-string
__doc__ = """
pca
=====================================================================

Description
-----------
pca: A Python Package for Principal Component Analysis.

Examples
--------
>>> from pca import pca
>>> # Load example data
>>> from sklearn.datasets import load_iris
>>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
>>>
>>> Initialize
>>> model = pca(n_components=3)
>>> # Fit using PCA
>>> results = model.fit_transform(X)
>>>
>>> # Make plots
>>> fig, ax = model.scatter()
>>>
>>> # Scree plot together with explained variance.
>>> fig, ax = model.plot()
>>>
>>> # Plot loadings
>>> fig, ax = model.biplot()
>>>
>>> # Plot loadings with outliers
>>> fig, ax = model.biplot(SPE=True, hotellingt2=True)
>>>
>>> 3D plots
>>> fig, ax = model.scatter3d()
>>> fig, ax = model.biplot3d()
>>> fig, ax = model.biplot3d(SPE=True, hotellingt2=True)
>>>
>>> # Normalize out PCs
>>> X_norm = model.norm(X)

References
----------
* Blog: https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559
* Github: https://github.com/erdogant/pca
* Documentation: https://erdogant.github.io/pca/

"""
