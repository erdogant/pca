from pca import pca
import pandas as pd

model = pca(normalize=True)
# Dataset
df = pd.read_csv('C://temp//usarrest.txt')
# Setup dataset
X = df[['Murder','Assault','UrbanPop','Rape']].astype(float)
X.index = df['state'].values

# Fit transform
out = model.fit_transform(X)
out['topfeat']

# Make plot
# ax = model.scatter(legend=False)
ax = model.biplot(n_feat=2, legend=False)
ax = model.biplot3d(n_feat=3, legend=False)


# %%
from sklearn.datasets import load_iris
import pandas as pd
from pca import pca

# Initialize
model = pca(n_components=3, normalize=True)
# Dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
# Fit transform
out = model.fit_transform(X)
out['topfeat']
# Make plot
ax = model.biplot(n_feat=1)
ax = model.biplot3d(n_feat=3)

# %%
import numpy as np
import pandas as pd
from pca import pca

f5=np.random.randint(0,100,250)
f2=np.random.randint(0,50,250)
f3=np.random.randint(0,25,250)
f4=np.random.randint(0,10,250)
f1=np.random.randint(0,5,250)
f6=np.random.randint(0,4,250)
f7=np.random.randint(0,3,250)
f8=np.random.randint(0,2,250)
f9=np.random.randint(0,1,250)
X = np.c_[f1,f2,f3,f4,f5,f6,f7,f8,f9]
X = pd.DataFrame(data=X, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])

# Initialize
model = pca()
# Fit transform
out = model.fit_transform(X)
out['topfeat']

model.plot()
ax = model.biplot(n_feat=10, legend=False)
ax = model.biplot3d(n_feat=10, legend=False)

# %% Normalize out PC1, PC2
X_norm = model.norm(X, pcexclude=[1,2])
X_norm = pd.DataFrame(data=X_norm, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
out = model.fit_transform(X_norm)
out['topfeat']

model.plot()
ax = model.biplot(n_feat=10, legend=False)

# %%
import pca
print(pca.__version__)

# %%
from sklearn.datasets import load_iris
import pandas as pd
from pca import pca

# Initialize
model = pca(n_components=3, normalize=True)

# Dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)

# Fit transform
out = model.fit_transform(X)

# Make plots
fig, ax = model.scatter()
ax = model.biplot(n_feat=4)
ax = model.plot()


ax = model.biplot2(n_feat=3)

# Make 3d plolts
model.scatter3d()
ax = model.biplot3d()

# Normalize out PCs
model = pca()
Xnew = model.norm(X)


# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

#In general it is a good idea to scale the data
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

pca = PCA()
pca.fit(X,y)
x_new = pca.transform(X)   

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
ax = model.biplot2(n_feat=100)

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
model1.fit_transform(X)
# len(np.unique(model1.results['topfeat'].iloc[:,1]))
model1.results['topfeat']

model1.plot()
model1.biplot(n_feat=10)
model1.biplot3d(n_feat=10)
model1.scatter()

# Initialize
model2 = pca(normalize=True, onehot=False)
# Run model 2
model2.fit_transform(X)
model2.plot()
model2.biplot(n_feat=4)
model2.scatter()
model2.biplot3d(n_feat=10)

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