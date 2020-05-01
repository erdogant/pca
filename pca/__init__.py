from pca.pca import (
    fit,
 	biplot,
 	biplot3d,
 	plot,
    norm,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.5'

# module level doc-string
__doc__ = """
pca
=====================================================================

Description
-----------
pca is a python package that performs the principal component analysis and to make insightful plots.

Examples
--------
>>> # Load example data
>>> X = load_iris().data
>>> labels = load_iris().feature_names
>>> y = load_iris().target
>>> Fit using PCA
>>> model = pca.fit(X, row_labels=y, col_labels=labels)
>>> ax = pca.biplot(model) 
>>> ax = pca.biplot3d(model)
>>> ax = pca.plot(model)
>>> X_norm = pca.norm(X)

References
----------
* https://github.com/erdogant/pca
* https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html

"""
