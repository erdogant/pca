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
out['outliers']

ax = model.scatter(legend=False)
ax = model.scatter3d(legend=False)

# Make plot
ax = model.biplot(n_feat=4, legend=False)
ax = model.biplot(n_feat=4, legend=False, label=False)

ax = model.biplot3d(n_feat=1, legend=False)
ax = model.biplot3d(n_feat=2, legend=False)
ax = model.biplot3d(n_feat=4, legend=False, label=False)

# %%
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy, random

# Generate data and fit PCA
# random.seed(1)
data = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
outliers = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)
data = np.vstack((data, outliers))
# pca = decomposition.PCA(n_components = 2)
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# pcaFit = pca.fit(data)
# dataProject = pcaFit.transform(data)

out = model.fit_transform(data)
data = model.results['PC'].values

#Calculate ellipse bounds and plot with scores
alpha = 0.05
nstd = 1
theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
circle = np.array((np.cos(theta), np.sin(theta)))


# z = -np.linspace(9,15,100)
# x = np.linspace(-26,26,1000)
# x,z = np.meshgrid(x,z)
# Z = -np.exp(-0.05*z) +4*(z+10)**2 
# X = x**2
# plt.contour(x, z, (X+Z), [0])

# Width and height are "full" widths, not radius
# from matplotlib.patches import Ellipse
# cov = np.cov(data[:, [0,1]], rowvar=False)
# vals, vecs = np.linalg.eigh(cov)
# width, height = 2 * nstd * np.sqrt(vals)
# ellip = Ellipse(xy=data[:, [0,1]], width=width, height=height, angle=theta)

# Covariance datapoints
sigma = np.cov(np.array((data[:, 0], data[:, 1])))
# sigma = np.cov(data)
# anomaly_score_threshold = np.sqrt(scipy.stats.chi2.ppf(1-alpha, nstd))
anomaly_score_threshold = scipy.stats.chi2.ppf(q=(1 - alpha), df=nstd)
ell = np.transpose(circle).dot(np.linalg.cholesky(sigma) * anomaly_score_threshold)

# anomaly_score = (new_x - np.mean(normal_x)) ** 2 / np.var(normal_x)
# anomaly_score = (new_x - np.mean(data)) ** 2 / np.var(data)
# out = anomaly_score > anomaly_score_threshold


# 95% ellipse bounds
a, b = np.max(ell[: ,0]), np.max(ell[: ,1])
t = np.linspace(0, 2 * np.pi, 100)


fig, ax = plt.subplots()
# ax.add_artist(ellip)
plt.scatter(data[:, 0], data[:, 1])
plt.plot(a * np.cos(t), b * np.sin(t), color = 'red')
plt.grid(color = 'lightgray', linestyle = '--')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

def plot_point_cov(data, nstd=2, calpha=0.05, ax=None, color='green'):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (data, an Nx2 array).

    Parameters
    ----------
        data : An Nx2 array of the data.
        nstd : The radius of the ellipse in numbers of standard deviations. Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    if ax is None:
        ax = plt.gca()

    # The 2x2 covariance matrix to base the ellipse on the location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
    data = data[:,[0,1]]
    pos = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)

    vals, vecs = eigsorted(cov, nstd)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    # ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, color=color, alpha=calpha)

    ax.add_artist(ellip)
    return ellip


def eigsorted(cov, nstd):
    vals, vecs = np.linalg.eigh(cov)
    # vecs = vecs * np.sqrt(scipy.stats.chi2.ppf(0.95, nstd))
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

# def multitest(P, weights):
#     return stats.combine_pvalues(P, method='stouffer', weights=weights)


def HotellingsT2(y, X, alpha=0.05, nstd=1):
    anomaly_score_threshold = stats.chi2.ppf(q=(1 - alpha), df=nstd)
    y_score = (y - np.mean(X)) ** 2 / np.var(X)
    y_proba = 1 - stats.chi2.cdf(y_score, df=nstd)
    y_bool = y_score >= anomaly_score_threshold
    
    Pcomb = []
    weights = np.arange(1/y_proba.shape[1], 1, 1/y_proba.shape[1])[::-1]
    for i in range(0,y_proba.shape[0]):
        Pcomb.append(stats.combine_pvalues(y_proba[i,:], method='stouffer', weights=weights))
        # stats.combine_pvalues(y_proba[-1,:], method='fisher')
    Pcomb = np.array(Pcomb)
    y_proba = Pcomb[:,1]
    y_score = Pcomb[:,0]
    y_bool =  Pcomb[:,1]<=alpha

    return y_proba, y_score, y_bool


#-- Example usage -----------------------
# Generate some random, correlated data
# points = np.random.multivariate_normal(mean=(1,1), cov=[[0.4, 9],[9, 10]], size=1000)
# points = model.results['PC'].values[:,[0,1]]

nstd=2
alpha=0.05

y_proba, y_score, y_bool = HotellingsT2(data, data, alpha=alpha, nstd=nstd)

# Plot the raw points...
# data = data[:,[0,1,2]]
plt.scatter(data[:,0], data[:,1], c='r')
Iloc = np.sum(y_bool,axis=1)
plt.scatter(data[Iloc>0,0], data[Iloc>0,1], c='g')
# Plot a transparent 3 standard deviation covariance ellipse
plot_point_cov(data, nstd=nstd, color='green', calpha=0.5, ax=None)
plt.show()

# %%

import numpy as np
from scipy.stats import f as f_distrib


def hotelling_t2(X, Y):
    
    # X and Y are 3D arrays
    # dim 0: number of features
    # dim 1: number of subjects
    # dim 2: number of mesh nodes or voxels (numer of tests)
    
    nx = X.shape[1]
    ny = Y.shape[1]
    p = X.shape[0]
    Xbar = X.mean(1)
    Ybar = Y.mean(1)
    Xbar = Xbar.reshape(Xbar.shape[0], 1, Xbar.shape[1])
    Ybar = Ybar.reshape(Ybar.shape[0], 1, Ybar.shape[1])
    
    X_Xbar = X - Xbar
    Y_Ybar = Y - Ybar
    Wx = np.einsum('ijk,ljk->ilk', X_Xbar, X_Xbar)
    Wy = np.einsum('ijk,ljk->ilk', Y_Ybar, Y_Ybar)
    W = (Wx + Wy) / float(nx + ny - 2)
    Xbar_minus_Ybar = Xbar - Ybar
    x = np.linalg.solve(W.transpose(2, 0, 1),
    Xbar_minus_Ybar.transpose(2, 0, 1))
    x = x.transpose(1, 2, 0)
    
    t2 = np.sum(Xbar_minus_Ybar * x, 0)
    t2 = t2 * float(nx * ny) / float(nx + ny)
    stat = (t2 * float(nx + ny - 1 - p) / (float(nx + ny - 2) * p))
    
    pval = 1 - np.squeeze(f_distrib.cdf(stat, p, nx + ny - 1 - p))
    return pval, t2

hout = hotelling_t2(data, data)

# %%
# import numpy as np
# from scipy import stats

# def HotellingsT2(new_x, normal_x, alpha=0.05):
#     anomaly_score_threshold = stats.chi2.ppf(q=(1 - alpha), df=1)
#     anomaly_score = (new_x - np.mean(normal_x)) ** 2 / np.var(normal_x)
#     out = anomaly_score > anomaly_score_threshold
#     return out, anomaly_score


# data1 = model.results['PC'].values
# outa = HotellingsT2(data1, data1, alpha=0.05)

# %%
from hnet import hnet
df = pd.read_csv('C://temp//usarrest.txt')
hn = hnet(y_min=3, perc_min_num=None)
results=hn.association_learning(df)
hn.plot()

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

# Note that the selected loading are relative to the PCs that are present.
# A weak loading can be larger then for example PC1 but that is because that specific feature showed relative weaker loading for PC1 and was therefore never selected.
out['topfeat']
# Make plot
ax = model.biplot(n_feat=1)
ax = model.biplot3d(n_feat=6)

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
ax = model.biplot(n_feat=2)
ax = model.plot()


ax = model.biplot3d(n_feat=2)

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