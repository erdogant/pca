from pca.pca import pca

from pca.pca import (
    import_example,
    )


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.3'

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
>>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
>>> # Fit using PCA
>>> results = model.fit_transform(X)
>>> # Make plots
>>> model.scatter()
>>> ax = model.plot()
>>> ax = model.biplot()
>>> 3D plots
>>> model.scatter3d()
>>> ax = model.biplot3d()
>>> # Normalize out PCs
>>> X_norm = pca.norm(X)

References
----------
* https://github.com/erdogant/pca
* https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html

"""
