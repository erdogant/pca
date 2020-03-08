# EXAMPLE
import pca
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV


# %% data
X = load_iris().data
labels=load_iris().feature_names
y=load_iris().target

# %%
param_grid = {
    'n_components':[None, 0.01, 1, 0.95, 2, 100000000000],
    'row_labels':[None, [], y],
    'col_labels':[None, [], labels],
    }

import itertools as it
allNames = param_grid.keys()
combinations = it.product(*(param_grid[Name] for Name in allNames))
combinations=list(combinations)

# %%
for combination in combinations:
    model = pca.fit(X, n_components=combination[0], row_labels=combination[1], col_labels=combination[2])
    ax = pca.plot(model)
    ax = pca.biplot(model)
    ax = pca.biplot3d(model)

# %%
import pca
from scipy.sparse import random as sparse_random
X = sparse_random(100, 1000, density=0.01, format='csr',random_state=42)

model = pca.fit(X)
ax = pca.plot(model)
ax = pca.biplot(model)
ax = pca.biplot3d(model)

# %%
X = load_iris().data
labels=load_iris().feature_names
y=load_iris().target

model = pca.fit(X)
ax = pca.plot(model)
ax = pca.biplot(model)
ax = pca.biplot3d(model)

model = pca.fit(X, row_labels=y, col_labels=labels)
fig = pca.biplot(model)
fig = pca.biplot3d(model)

model = pca.fit(X, n_components=0.95)
ax = pca.plot(model)
ax   = pca.biplot(model)

model = pca.fit(X, n_components=2)
ax = pca.plot(model)
ax   = pca.biplot(model)

Xnorm = pca.norm(X, pcexclude=[1,2])
model = pca.fit(Xnorm, row_labels=y, col_labels=labels)
ax = pca.biplot(model)
ax = pca.plot(model)