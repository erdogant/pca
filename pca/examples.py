from pca import pca
import pandas as pd
import numpy as np

X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
outliers = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)
X = np.vstack((X, outliers))

model = pca(alpha=0.05, n_std=2)

# Fit transform
out = model.fit_transform(X)
out['topfeat']
print(out['outliers'])

model.biplot(legend=True, SPE=True, hotellingt2=True)
model.biplot(legend=True, SPE=True, hotellingt2=False)
model.biplot(legend=True, SPE=False, hotellingt2=True)
model.biplot(legend=True, SPE=False, hotellingt2=False)

model.biplot3d(legend=True, SPE=True, hotellingt2=True)
model.biplot3d(legend=True, SPE=True, hotellingt2=False)
model.biplot3d(legend=True, SPE=False, hotellingt2=True)
model.biplot3d(legend=True, SPE=False, hotellingt2=False)

model.scatter(legend=True, SPE=True, hotellingt2=True)
model.scatter(legend=True, SPE=True, hotellingt2=False)
model.scatter(legend=True, SPE=False, hotellingt2=True)
model.scatter(legend=True, SPE=False, hotellingt2=False)

model.scatter3d(legend=True, SPE=True, hotellingt2=True)
model.scatter3d(legend=True, SPE=True, hotellingt2=False)
model.scatter3d(legend=True, SPE=False, hotellingt2=True)
model.scatter3d(legend=True, SPE=False, hotellingt2=False)


ax = model.biplot(n_feat=4, legend=False, label=False)

import pca
outliers_hot = pca.hotellingsT2(out['PC'].values, alpha=0.05)
outliers_spe = pca.spe_dmodx(out['PC'].values, n_std=2)

# Select the outliers
Xoutliers = X[out['outliers']['y_bool'],:]

# Select the other set
Xnormal = X[~out['outliers']['y_bool'],:]

import pca
outliers, ax = pca.compute_elipse_and_outliers(out['PC'].values, nstd=2, color='green', calpha=0.5, showfig=True)


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

# def _eigsorted(cov, nstd):
#     vals, vecs = np.linalg.eigh(cov)
#     # vecs = vecs * np.sqrt(scipy.stats.chi2.ppf(0.95, nstd))
#     order = vals.argsort()[::-1]
#     return vals[order], vecs[:,order]


# def compute_elipse_and_outliers(X, nstd=2, calpha=0.5, ax=None, color='green', showfig=False):
#     """Plot ellipse based on the mean and covariance of the first 2 dimensions of X.

#     Parameters
#     ----------
#     X : Array-like
#         Input data, in this case the Principal components.
#     nstd : int, (default: 2)
#         Standard deviation. The default is 2.
#     calpha : float, (default: 0.05)
#         transperancy color.
#     color : String, (default: 'green')
#         Color of the ellipse.
#     ax : object, (default : None)
#         Figure axis.
#     showfig : bool, (default : False)
#         Show figure.

#     Returns
#     -------
#     outliers : pd.DataFrame()
#         column with boolean outliers and euclidean distance of each sample to the center of the ellipse.
#     ax : object
#         Figure axis.

#     """
#     # The 2x2 covariance matrix to base the ellipse on the location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
#     X = X[:,[0,1]]
#     # Compute mean and covariance
#     g_ell_center = X.mean(axis=0)
#     cov = np.cov(X, rowvar=False)
#     # Width and height are "full" widths, not radius
#     vals, vecs = _eigsorted(cov, nstd)
#     angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
#     width, height = 2 * nstd * np.sqrt(vals)
#     # g_ellipse  = Ellipse(xy=g_ell_center, width=width, height=height, angle=angle, color=color, alpha=calpha)
#     # Compute angles of ellipse
#     cos_angle = np.cos(np.radians(180.-angle))
#     sin_angle = np.sin(np.radians(180.-angle))
#     # Determine the elipse range    
#     xc = X[:,0] - g_ell_center[0]
#     yc = X[:,1] - g_ell_center[1]
#     xct = xc * cos_angle - yc * sin_angle
#     yct = xc * sin_angle + yc * cos_angle 
#     rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)
    
#     from sklearn.metrics.pairwise import euclidean_distances
#     g_ell_center
    
#     # Mark the samples within or outside the ellipse
#     outliers = []
#     for r in rad_cc:
#         if r <= 1.:
#             # point in ellipse
#             outliers.append(False)
#         else:
#             # point not in ellipse
#             outliers.append(True)

#     # Plot the raw points.
#     if showfig:
#         if ax is None: ax = plt.gca()
#         g_ellipse  = Ellipse(xy=g_ell_center, width=width, height=height, angle=angle, color=color, alpha=calpha)
#         ax.add_artist(g_ellipse)
#         ax.scatter(X[:,0],  X[:,1], c=outliers, linewidths=0.3)
    
#     out = pd.DataFrame()
#     out['outliers'] = outliers
#     out['eucl_score'] =  list(map(lambda x: euclidean_distances([g_ell_center], x.reshape(1,-1))[0][0], X))
    
#     return out, ax


nstd=2
# Plot a transparent 3 standard deviation covariance ellipse
outliers, ax = compute_elipse_and_outliers(X, nstd=nstd, color='green', calpha=0.5, showfig=True)

# %%
from hnet import hnet
df = pd.read_csv('C://temp//usarrest.txt')
hn = hnet(y_min=3, perc_min_num=None)
results=hn.association_learning(df)
hn.plot()

# %%

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

ax = model.scatter(legend=False, )
ax = model.scatter3d(legend=False)

# Make plot
ax = model.biplot(n_feat=4, legend=False)
ax = model.biplot(n_feat=4, legend=False, label=False)

ax = model.biplot3d(n_feat=1, legend=False)
ax = model.biplot3d(n_feat=2, legend=False)
ax = model.biplot3d(n_feat=4, legend=False, label=False)

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
out['outliers']
# Make plot
model.plot()
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

# %% IRIS DATASET EXAMPLE
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
model1.results['outliers']

model1.plot()
model1.biplot(n_feat=10)
model1.biplot3d(n_feat=10)
model1.scatter()
model1.scatter3d()

import pca
pca.hotellingsT2(model1.results['PC'].values, model1.results['PC'].values)

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