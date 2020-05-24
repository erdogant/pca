
# %%
import pca
print(pca.__version__)

# %%
from sklearn.datasets import load_iris
import pandas as pd
from pca import pca

# Initialize
model = pca(n_components=3)

# Dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)

# Fit transform
out = model.fit_transform(X)

# Make plots
model.scatter()
ax = model.biplot(n_feat=4)
ax = model.plot()

# Make 3d plolts
model.scatter3d()
ax = model.biplot3d()

# Normalize out PCs
model = pca()
Xnew = model.norm(X)





# %%
X = pd.read_csv('D://GITLAB/MASTERCLASS/embeddings/data/TCGA_RAW.zip',compression='zip')
metadata = pd.read_csv('D://GITLAB/MASTERCLASS/embeddings/data/metadata.csv', sep=';')
features = pd.read_csv('D://GITLAB/MASTERCLASS/embeddings/data/features.csv')
X = pd.DataFrame(data=X.values, index=features.values.flatten(), columns=metadata.labx.values).T

# %% RAW data
# Initializatie
model = pca(n_components=0.95, normalize=True)
# Fit transform
results = model.fit_transform(X)

# Make plots
model.scatter()
ax = model.plot()
ax = model.biplot(n_feat=20)

# %%
import numpy as np
from tqdm import tqdm
from pca import pca
import matplotlib.pyplot as plt

# Normalize
Xnorm = np.log2(X+1)
rowmeans = np.mean(Xnorm, axis=0)
for i in tqdm(range(Xnorm.shape[1])):
    Xnorm.iloc[:,i] = Xnorm.values[:,i] - rowmeans[i]

# Make histogram
plt.hist(Xnorm.values.ravel(), bins=50)

# Initializatie
model = pca(n_components=0.95, normalize=False)
# Fit transform
results = model.fit_transform(Xnorm)

# Make plots
model.scatter()
ax = model.plot()
from pca import pca
ax = model.biplot(n_feat=100)

model.scatter3d()
ax = model.biplot3d(n_feat=20)

# %% Exmample with mixed dataset
import pca
# Import example
df = pca.import_example()

# Transform data into one-hot
from df2onehot import df2onehot
y = df['Survived'].values
del df['Survived']
del df['PassengerId']
del df['Name']
out = df2onehot(df)
X = out['onehot'].copy()
X.index = y


from pca import pca

# Initialize
model1 = pca(normalize=False, onehot=False)
# Run model 1
_=model1.fit_transform(X)
model1.plot()
model1.biplot(n_feat=3)
model1.scatter()
model1.biplot3d(n_feat=3)

# Initialize
model2 = pca(normalize=True, onehot=False)
# Run model 2
_=model2.fit_transform(X)
model2.plot()
model2.biplot(n_feat=4)
model2.scatter()
model2.biplot3d(n_feat=3)

# Initialize
model3 = pca(normalize=False, onehot=True)
# Run model 2
_=model3.fit_transform(X)
model3.biplot(n_feat=3)

# %%
# # EXAMPLE
# import pca
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import GridSearchCV


# # %% data
# X = load_iris().data
# labels=load_iris().feature_names
# y=load_iris().target

# # %%
# param_grid = {
#     'n_components':[None, 0.01, 1, 0.95, 2, 100000000000],
#     'row_labels':[None, [], y],
#     'col_labels':[None, [], labels],
#     }

# import itertools as it
# allNames = param_grid.keys()
# combinations = it.product(*(param_grid[Name] for Name in allNames))
# combinations=list(combinations)

# # %%
# for combination in combinations:
#     model = pca.fit(X, n_components=combination[0], row_labels=combination[1], col_labels=combination[2])
#     ax = pca.plot(model)
#     ax = pca.biplot(model)
#     ax = pca.biplot3d(model)

# # %%
# import pca
# from scipy.sparse import random as sparse_random
# X = sparse_random(100, 1000, density=0.01, format='csr',random_state=42)

# model = pca.fit(X)
# ax = pca.plot(model)
# ax = pca.biplot(model)
# ax = pca.biplot3d(model)

# # %%
# import pandas as pd
# X = load_iris().data
# labels=load_iris().feature_names
# y=load_iris().target

# df = pd.DataFrame(data=X, columns=labels)
# model = pca.fit(df)


# # %%
# X = load_iris().data
# labels=load_iris().feature_names
# y=load_iris().target

# model = pca.fit(X)
# ax = pca.plot(model)
# ax = pca.biplot(model)
# ax = pca.biplot3d(model)

# model = pca.fit(X, row_labels=y, col_labels=labels)
# fig = pca.biplot(model)
# fig = pca.biplot3d(model)

# model = pca.fit(X, n_components=0.95)
# ax = pca.plot(model)
# ax   = pca.biplot(model)

# model = pca.fit(X, n_components=2)
# ax = pca.plot(model)
# ax   = pca.biplot(model)

# Xnorm = pca.norm(X, pcexclude=[1,2])
# model = pca.fit(Xnorm, row_labels=y, col_labels=labels)
# ax = pca.biplot(model)
# ax = pca.plot(model)