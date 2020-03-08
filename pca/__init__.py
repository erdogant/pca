from pca.pca import (
    fit,
 	biplot,
 	biplot3d,
 	plot,
    norm,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.4'

# module level doc-string
__doc__ = """
pca - pca is a python package that performs the principal component analysis and to make insightful plots.
=====================================================================

Description
-----------
Principal Component Analysis and insightful plots.

Example
-------
>>> model = pca.fit(X)
>>> ax = pca.biplot(model) 
>>> ax = pca.biplot3d(model)
>>> ax = pca.plot(model)
>>> X_norm = pca.norm(X)

"""
