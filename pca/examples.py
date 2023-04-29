from pca import pca
import pandas as pd
import numpy as np
import matplotlib as mpl
from scatterd import scatterd

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

# %%
# Load pca
from pca import pca

# Initialize pca
model = pca(n_components=3)

# Load example data set
df = model.import_example(data='iris')

# Fit transform
results = model.fit_transform(df)

# Make plot
model.biplot(HT2=True,
             SPE=True,
             s=np.random.randint(20, 500, size=df.shape[0]),
             marker=df.index.values,
             cmap='bwr_r',
             fontsize=22,
             legend=2,
             density=True,
             arrowdict={'color_strong': 'r', 'color_weak': 'g'},
             title='Biplot with with the pca library.')

# model.biplot(arrowdict={'color_strong': 'r', 'color_weak': 'g'})

# %%
# from sklearn.datasets import make_friedman1
# X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)

# model = pca()
# model.fit_transform(X)

# Loading are automatically set based on weak/strong
model.biplot(s=0)
# Change strong and weak colors
model.biplot(s=0, arrowdict={'color_strong': 'r', 'color_weak': 'g'})
# Set alpha to constant value
model.biplot(s=0, arrowdict={'alpha': 0.8})
# Change arrow text color
model.biplot(s=0, arrowdict={'color_text': 'k'})
# Change arrow color, which automatically changes the label color too
model.biplot(s=0, color_arrow='k')
# Almost Remove arrows but keep the text
model.biplot(s=0, color_arrow='k', arrowdict={'alpha': 0.9})
# Set color text
model.biplot(s=0, arrowdict={'color_text': 'k'})
# Set color arrow and color text
model.biplot(s=0, color_arrow='k', arrowdict={'color_text': 'g'})
# Set color arrow and color text and alpha
model.biplot(s=0, color_arrow='k', arrowdict={'color_text': 'g', 'alpha': 0.8})

# Change the scale factor of the arrow
model.biplot(arrowdict={'scale_factor': 2})
model.biplot3d(arrowdict={'scale_factor': 3})

model.biplot(s=0, arrowdict={'weight':'bold', 'fontsize': 24, 'color_text': 'r'}, color_arrow='k')
model.biplot3d(density=True, fontsize=0, arrowdict={'weight':'bold', 'fontsize': 14})
model.biplot3d(density=True, fontsize=0, s=0, arrowdict={'fontsize': 24})
model.biplot3d(density=True, fontsize=0, s=0, arrowdict={'fontsize': 24, 'scale_factor': 3})
model.biplot(density=True, fontsize=0, arrowdict={'weight':'bold', 'fontsize': 14})


# %%
from pca import pca
from sklearn.datasets import load_wine
import pandas as pd

# Load dataset
data = load_wine()
X = data.data
y = data.target
labels = data.feature_names
# Make dataframe
df = pd.DataFrame(data=X, columns=labels, index=y)

model = pca(normalize=True, n_components=None)
# Fit transform with dataframe
results = model.fit_transform(df)
# Plot
model.biplot(fontsize=0, labels=df['flavanoids'].values, legend=False, cmap='seismic', n_feat=3, arrowdict={'fontsize':28, 'c':'g'}, density=True)
# model.biplot3d(legend=None, n_feat=3, fontcolor='r', arrowdict={'fontsize':22, 'c':'k'}, density=True, figsize=(35, 30))


# %% Demonstration of specifying colors, markers, alpha, and size per sample
# Import library
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_friedman1
from pca import pca

# Make data set
X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)

# All available markers
markers = np.array(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
# Create colors
cmap = plt.cm.get_cmap('tab20c', len(markers))
# Generate random integers
random_integers = np.random.randint(0, len(markers), size=X.shape[0])
# Draw markers
marker = markers[random_integers]
# Set colors
color = cmap.colors[random_integers, 0:3]
# Set Size
size = np.random.randint(50, 1000, size=X.shape[0])
# Set alpha
alpha = np.random.rand(1, X.shape[0])[0][random_integers]

# Init
model = pca(verbose=3)
# Fit
model.fit_transform(X)
# Make plot with blue arrows and text
fig, ax = model.biplot(
                        SPE=True,
                        HT2=True,
                        c=color,
                        s=size,
                        marker=marker,
                        alpha=0.4,
                        color_arrow='k',
                        title='Demonstration of specifying colors, markers, alpha, and size per sample.',
                        n_feat=5,
                        fontsize=20,
                        fontweight='normal',
                        arrowdict={'fontsize': 18},
                        density=True,
                        density_on_top=False,
                        )


# %%
from df2onehot import df2onehot
from pca import pca

model = pca()
df = model.import_example(data='titanic')

# Create one-hot array
df_hot = df2onehot(df,
                   remove_mutual_exclusive=True,
                   excl_background=['PassengerId', 'None'],
                   y_min=10,
                   verbose=4)['onehot']

# Initialize
model = pca(normalize=True, detect_outliers=['HT2', 'SPE'])

# Fit
model.fit_transform(df_hot)

# model.scatter(legend=False)
# model.biplot(legend=False)

model.biplot(SPE=True,
              HT2=True,
              marker=df['Survived'],
              s=df['Age']*20,
              n_feat=2,
              labels=df['Sex'],
              title='Biplot with with the pca library.',
              color_arrow='k',
              fontsize=28,
              fontcolor=None,
              arrowdict={'fontsize': 18},
              cmap='bwr_r',
              edgecolor='#FFFFFF',
              gradient='#FFFFFF',
              density=True,
              density_on_top=False,
              visible=True,
              )

# %%
# Load pca
from pca import pca

# Initialize pca
model = pca(n_components=3)

# Load example data set
df = model.import_example(data='iris')

# Fit transform
results = model.fit_transform(df)

# Make plot
model.biplot(HT2=True,
             SPE=True,
             s=np.random.randint(20, 500, size=df.shape[0]),
             marker=y,
             cmap='bwr_r',
             fontsize=22,
             legend=2,
             density=True,
             title='Biplot with with the pca library.')

# %%
df = pd.read_pickle('WIM-data PCA bug')
df = df.iloc[0:10000, :]
X = df.drop(['SUBCATEGORIE','UTCPASSAGEDATUM','STROOKVOLGNUMMER','TWEESTROKEN'],axis=1)
y = df['SUBCATEGORIE']
model = pca(normalize=True, n_components=0.95)
results = model.fit_transform(X, col_labels=X.columns, row_labels=y)

# results = model.fit_transform(X)
model.scatter3d()
model.biplot3d()


model.scatter()
model.scatter(fontsize=20, c=None)
model.scatter(fontsize=20, cmap=None)
model.scatter(edgecolor='#FFFFFF')
model.scatter(edgecolor='#FFFFFF', gradient='#FFFFFF')
model.scatter(density=True)
model.scatter(density=True, c=None)
model.scatter(density=True, c=None, legend=False)
model.scatter(density=True, c=None, fontsize=0)
model.scatter(density=True, s=10, edgecolor=None, fontsize=0, alpha=0.2)

model.scatter(fontsize=20)
model.biplot()
model.scatter(c=[0,0,0], legend=True)
model.scatter(alpha=None)
model.scatter(alpha=0.8)
model.scatter(alpha=0.8, density=True)
model.scatter(s=250, alpha=0.8, gradient='#FFFFFF', edgecolor=None)

model.biplot()
model.biplot(cmap=None, n_feat=5)
model.biplot3d()
model.biplot3d(cmap=None, n_feat=8)






# %% Test examples
import pca
# Import example
df = pca.import_example('sprinkler')
df = pca.import_example('titanic')
df = pca.import_example('student')

# %% change marker
# Import library
import numpy as np
from sklearn.datasets import make_friedman1
from pca import pca

# Make data set
X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)

# All available markers
markers = np.array(['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
# Generate random integers
random_integers = np.random.randint(0, len(markers), size=X.shape[0])
# Draw markers
marker = markers[random_integers]

# Init
model = pca(verbose=3)
# Fit
model.fit_transform(X)
# Make plot with blue arrows and text
fig, ax = model.biplot(c=[0, 0, 0],
                       marker=marker,
                       title='Demonstration of specifying colors, markers, alpha, and size per sample.',
                       n_feat=5,
                       legend=None)




# %%


# Import library
from pca import pca

# Initialize
model = pca()

# Load Titanic data set
df = model.import_example(data='student')

#     school sex  age address famsize Pstatus  ...  Walc  health absences  G1  G2  G3
# 0       GP   F   18       U     GT3       A  ...     1       3        4   0  11  11
# 1       GP   F   17       U     GT3       T  ...     1       3        2   9  11  11
# 2       GP   F   15       U     LE3       T  ...     3       3        6  12  13  12
# 3       GP   F   15       U     GT3       T  ...     1       5        0  14  14  14
# 4       GP   F   16       U     GT3       T  ...     2       5        0  11  13  13
# ..     ...  ..  ...     ...     ...     ...  ...   ...     ...      ...  ..  ..  ..
# 644     MS   F   19       R     GT3       T  ...     2       5        4  10  11  10
# 645     MS   F   18       U     LE3       T  ...     1       1        4  15  15  16
# 646     MS   F   18       U     GT3       T  ...     1       5        6  11  12   9
# 647     MS   M   17       U     LE3       T  ...     4       2        6  10  10  10
# 648     MS   M   18       R     LE3       T  ...     4       5        4  10  11  11

# [649 rows x 33 columns]

# Initialize
from df2onehot import df2onehot

df_hot = df2onehot(df)['onehot']

print(df_hot)


#      school_GP  school_MS  sex_F  sex_M  ...  G3_6.0  G3_7.0  G3_8.0  G3_9.0
# 0         True      False   True  False  ...   False   False   False   False
# 1         True      False   True  False  ...   False   False   False   False
# 2         True      False   True  False  ...   False   False   False   False
# 3         True      False   True  False  ...   False   False   False   False
# 4         True      False   True  False  ...   False   False   False   False
# ..         ...        ...    ...    ...  ...     ...     ...     ...     ...
# 644      False       True   True  False  ...   False   False   False   False
# 645      False       True   True  False  ...   False   False   False   False
# 646      False       True   True  False  ...   False   False   False    True
# 647      False       True  False   True  ...   False   False   False   False
# 648      False       True  False   True  ...   False   False   False   False

# [649 rows x 177 columns]

model = pca(normalize=True,
            detect_outliers=['HT2', 'SPE'],
            alpha=0.05,
            n_std=3,
            multipletests='fdr_bh')

results = model.fit_transform(df_hot)

overlapping_outliers = np.logical_and(results['outliers']['y_bool'], results['outliers']['y_bool_spe'])
df.loc[overlapping_outliers]

#     school sex  age address famsize Pstatus  ...  Walc  health absences G1 G2 G3
# 279     GP   M   22       U     GT3       T  ...     5       1       12  7  8  5
# 284     GP   M   18       U     GT3       T  ...     5       5        4  7  8  6
# 523     MS   M   18       U     LE3       T  ...     5       5        2  5  6  6
# 605     MS   F   19       U     GT3       T  ...     3       2        0  5  0  0
# 610     MS   F   19       R     GT3       A  ...     4       1        0  8  0  0

# [5 rows x 33 columns]


# Make biplot
model.biplot(SPE=True,
             HT2=True,
             jitter=0.1,
             n_feat=10,
             legend=False,
             labels=df['sex'],
             title='Student Performance',
             figsize=(20, 12),
             color_arrow='k',
             cmap='bwr_r',
             gradient='#FFFFFF',
             density=True,
             )


# %%
# Import library
from pca import pca
from sklearn.datasets import make_friedman1
X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)

# Init
model = pca()
# Fit
model.fit_transform(X)

# Make plot with blue arrows and text
fig, ax = model.biplot(c=[0,0,0], fontsize=20, color_arrow='blue', title=None, HT2=True, n_feat=10, visible=True)

# Use the existing fig and create new edits such red arrows for the first three loadings. Also change the font sizes.
fig, ax = model.biplot(c=[0,0,0], fontsize=20, arrowdict={'weight':'bold'}, color_arrow='red', n_feat=3, title='updated fig.', visible=True, fig=fig)


# %%
# https://github.com/erdogant/pca/issues/40

# Import iris dataset and other required libraries
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib as mpl
import colourmap
from pca import pca

# Class labels
y = load_iris().target

# Initialize pca
model = pca(n_components=3, normalize=True)
# Dataset
X = pd.DataFrame(index=y, data=load_iris().data, columns=load_iris().feature_names)
# Fit transform
out = model.fit_transform(X)

# The default setting is to color on classlabels (y). These are provided as the index in the dataframe.
model.biplot()

# Use custom cmap for classlabels (as an example I explicitely provide three colors).
model.biplot(cmap=mpl.colors.ListedColormap(['green', 'red', 'blue']))

# Set custom classlabels. Coloring is based on the input colormap (cmap).
y[10:15]=4
model.biplot(labels=y, cmap='Set2')

# Set custom classlabels and also use custom colors.
c = colourmap.fromlist(y, cmap='Set2')[0]
c[10:15] = [0,0,0]
model.biplot(labels=y, c=c)


# Remove scatterpoints by setting cmap=None
model.biplot(cmap=None)


# Color on classlabel (Unchanged)
model.biplot()
# Use cmap colors for classlabels (unchanged)
model.biplot(labels=y, cmap=mpl.colors.ListedColormap(['green', 'red', 'blue']), density=True)
# Do not show points when cmap=None (unchanged)
model.biplot(labels=load_iris().target, cmap=None, density=True)
# Plot all points as unique entity (unchanged)
model.biplot(labels=y, gradient='#ffffff', cmap=mpl.colors.ListedColormap(['green', 'red', 'blue']), density=True)


# %%
import numpy as np
from sklearn.datasets import load_iris

# Load dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)

# Load pca
from pca import pca
model = pca(n_components=0.95)
model = pca(n_components=3)
results = model.fit_transform(X, verbose=2)


model.scatter()
model.scatter(gradient='#ffffff')
model.scatter(gradient='#ffffff', cmap='tab20')
model.scatter(gradient='#ffffff', cmap='tab20', labels=np.ones_like(model.results['PC'].index.values))

model.scatter(gradient='#5dfa02')
model.scatter(gradient='#5dfa02', cmap='tab20')
model.scatter(gradient='#5dfa02', cmap='tab20', labels=np.ones_like(model.results['PC'].index.values))


# %%
# import pca
# print(pca.__version__)

# np.random.seed(0)
# x, y = np.random.random((2,30))
# fig, ax = plt.subplots()
# plt.plot(x, y, 'bo')
# texts = [plt.text(x[i], y[i], 'Text%s' %i) for i in range(len(x))]
# adjust_text(texts)



# %% Detect outliers in new unseen data.
# Import libraries
from pca import pca
import pandas as pd
import numpy as np

# Create dataset with 100 samples
X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)

# Initialize model. Alpha is the threshold for the hotellings T2 test to determine outliers in the data.
model = pca(alpha=0.05, detect_outliers=['HT2', 'SPE'])

# Fit transform
model.fit_transform(X)

# Create 5 outliers
X_unseen = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)

# map the new "unseen" data in the existing space
PCnew = model.transform(X_unseen)

# Plot image
model.biplot(SPE=True, HT2=True, density=True)
model.biplot3d(SPE=True, HT2=True, density=True, arrowdict={'scale_factor': 1})

# %% Detect unseen outliers
# Import libraries
from pca import pca
import pandas as pd
import numpy as np

# Create dataset with 100 samples
X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)

# Initialize model. Alpha is the threshold for the hotellings T2 test to determine outliers in the data.
model = pca(alpha=0.05, detect_outliers=['HT2', 'SPE'])
# model = pca(alpha=0.05, detect_outliers=None)

# Fit transform
model.fit_transform(X, row_labels=np.zeros(X.shape[0]))

model.scatter(SPE=True, HT2=True)

for i in range(0, 10):
    # Create 5 outliers
    X_unseen = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)

    # Transform new "unseen" data.
    PCnew = model.transform(X_unseen, row_labels=np.repeat('mapped_' + str(i), X_unseen.shape[0]), update_outlier_params=True)

    # Scatterplot
    # model.scatter(SPE=True, HT2=True)
    # Biplot
    model.biplot(SPE=True, HT2=True, density=True)


# %%
from sklearn.datasets import make_friedman1
X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)

model = pca(method='sparse_pca')
model = pca(method='trunc_svd')
model.fit_transform(X)
model.plot()
model.biplot()
model.biplot3d()
model.scatter()

# %%
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
np.random.seed(0)
X_dense = np.random.rand(100, 100)
X_dense[:, 2 * np.arange(50)] = 0
X = csr_matrix(X_dense)

# svd = TruncatedSVD(n_components=2)
# svd.fit(S)
model3 = pca(method='trunc_svd')
model3.fit_transform(X)
model3.biplot(n_feat=3)



# %%
from pca import pca
from sklearn.datasets import load_iris

index_name = load_iris().target.astype(str)
index_name[index_name=='0'] = load_iris().target_names[0]
index_name[index_name=='1'] = load_iris().target_names[1]
index_name[index_name=='2'] = load_iris().target_names[2]

X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=index_name)
model = pca(n_components=3, normalize=True)

out = model.fit_transform(X)
# fig, ax = model.biplot(legend=False, PC=[1,0])
print(out['topfeat'])

fig, ax = model.biplot(cmap='Set1', legend=True)
fig, ax = model.biplot(cmap='Set1', legend=False)
fig, ax = model.biplot(cmap='Set1', legend=False)
fig, ax = model.biplot(cmap=None, legend=False)
fig, ax = model.biplot(cmap=None, legend=True)
fig, ax = model.biplot(cmap=None, legend=True)

# %%
import numpy as np
from sklearn.datasets import load_iris

# Load dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)

# Load pca
from pca import pca

# Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
model = pca(n_components=0.95)

# Reduce the data towards 3 PCs
model = pca(n_components=3)

# Fit transform
results = model.fit_transform(X)

# model.plot()

model.scatter(gradient='#ffffff', cmap='Set1')
model.scatter(cmap='Set1', legend=True, gradient='#ffffff')
model.scatter(cmap='Set1', legend=True, gradient=None)

model.scatter3d(cmap='Set1', legend=True, gradient='#ffffff')
model.scatter3d(cmap='Set1', legend=True, gradient=None)

model.biplot(legend=True, SPE=True, HT2=True, visible=True, gradient='#ffffff')
model.biplot(legend=True, SPE=True, HT2=True, visible=True, gradient=None)
model.biplot3d(legend=True, SPE=True, HT2=True, visible=True, gradient=None)
model.biplot3d(legend=True, SPE=True, HT2=True, visible=True, gradient='#ffffff')


# %%
from pca import pca
from sklearn.datasets import load_iris
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
model = pca(n_components=3, normalize=True)

out = model.fit_transform(X)
# fig, ax = model.biplot(legend=False, PC=[1,0])
print(out['topfeat'])

fig, ax = model.biplot(cmap='Set1', legend=True)
fig, ax = model.biplot(cmap='Set1', legend=False)
fig, ax = model.biplot(cmap='Set1', legend=False)
fig, ax = model.biplot(cmap=None, legend=False)
fig, ax = model.biplot(cmap=None, legend=True)
fig, ax = model.biplot(cmap=None, legend=True)


# %%
import numpy as np
import pandas as pd
from pca import pca

feat_1 = np.random.randint(0,100,250)
feat_2 = np.random.randint(0,50,250)
feat_3 = np.random.randint(0,25,250)
feat_4 = np.random.randint(0,10,250)
feat_5 = np.random.randint(0,5,250)
feat_6 = np.random.randint(0,4,250)
feat_7 = np.random.randint(0,3,250)
feat_8 = np.random.randint(0,1,250)

# Make dataset
X = np.c_[feat_1, feat_2, feat_3, feat_4, feat_5, feat_6 ,feat_7, feat_8]
X = pd.DataFrame(data=X, columns=['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8'])

# fig, ax = plt.subplots(figsize=(20, 12))
# X = np.c_[f8,f7,f6,f5,f4,f3,f2,f1]
# X = pd.DataFrame(data=X, columns=['feat_8','feat_7','feat_6','feat_5','feat_4','feat_3','feat_2','feat_1'])
# X.plot.hist(bins=50, cmap='Set1', ax=ax)
# ax.grid(True)
# ax.set_xlabel('Value')

# Initialize
model = pca(n_components=None, normalize=False)
# Fit transform data
results = model.fit_transform(X)
# Extract the most informative features
results['topfeat']

#     PC feature   loading  type
# 0  PC1  feat_1 -0.997830  best
# 1  PC2  feat_2 -0.997603  best
# 2  PC3  feat_3  0.998457  best
# 3  PC4  feat_4  0.997536  best
# 4  PC5  feat_5 -0.952390  best
# 5  PC6  feat_6 -0.955873  best
# 6  PC7  feat_7 -0.994602  best
# 7  PC1  feat_8 -0.000000  weak

# Plot the explained variance
model.plot()
# Biplot with the loadings
ax = model.biplot(legend=False)
# Biplot with the loadings
ax = model.biplot(n_feat=3, legend=False, )
# Cleaning the biplot by removing the scatter, and looking only at the top 3 features.
ax = model.biplot(n_feat=3, legend=False, cmap=None)
# Make plot with 3 dimensions
model.biplot3d(n_feat=3, legend=False, cmap=None)

ax = model.biplot(n_feat=3, legend=False, cmap=None, PC=[1,2])
ax = model.biplot(n_feat=3, legend=False, cmap=None, PC=[2,3])


ax = model.biplot(n_feat=10, legend=False, cmap=None, )
ax = model.biplot(n_feat=10, legend=False, PC=[0, 1], )
ax = model.biplot(n_feat=10, legend=False, PC=[1, 0])
ax = model.biplot(n_feat=10, legend=False, PC=[0, 1, 2], d3=True)
ax = model.biplot(n_feat=10, legend=False, PC=[2, 1, 0], d3=True)
ax = model.biplot(n_feat=10, legend=False, PC=[2, 0, 1], d3=True)
ax = model.biplot(n_feat=10, legend=False, PC=[0, 1])
ax = model.biplot(n_feat=10, legend=False, PC=[0, 2])
ax = model.biplot(n_feat=10, legend=False, PC=[2, 1])
ax = model.biplot3d(n_feat=10, legend=False)

# model.scatter(labels=X.index.values==0)

# %%
from pca import pca

# Initialize
model = pca(alpha=0.05, n_std=2)

n_total = 10000
n_features = 10
X = np.random.randint(low=1, high=10, size=(n_total, n_features))

# Example with Numpy array
row_labels = np.arange(0, X.shape[0]).astype(str)
# Fit transform
out = model.fit_transform(X, row_labels=row_labels)
# Make plot
model.biplot(legend=False, PC=[0, 1], )

# %%
# Example with DataFrame
X = pd.DataFrame(data=X, columns=np.arange(0, X.shape[1]).astype(str))
# Fit transform
out = model.fit_transform(X)
# Make plot
model.biplot(legend=False, PC=[1, 2])
model.biplot(legend=False, PC=[1, 2, 3], d3=True)
model.biplot3d(legend=False, PC=[0, 1, 2])


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pca import pca

np.random.seed(42)
# Load dataset
n_total, train_ratio = 10000, 0.8
n_features = 10
my_array = np.random.randint(low=1, high=10, size=(n_total, n_features))
features = [f'f{i}' for i in range(1, n_features+1, 1)]
X = pd.DataFrame(my_array, columns=features)
X_train = X.sample(frac=train_ratio)
X_test = X.drop(X_train.index)

# Training
model = pca(n_components=5, alpha=0.5, n_std=3, normalize=True, random_state=42)
results = model.fit_transform(X=X_train[features])

# Inference: mapping of data into space.
PC_test = model.transform(X=X_test[features])
# Compute new outliers
scores, _ = model.compute_outliers(PC=PC_test, n_std=3, verbose=3) 

# Prepare for plotting
T2_train = np.log(results['outliers']['y_score'])
T2_mu, T2_sigma = T2_train.agg(['mean', 'std'])
T2_limit = T2_mu + T2_sigma*3
T2_test = np.log(scores['y_score'])

# Plot
plt.figure(figsize=(14, 4))
plt.axhline(T2_mu, color='blue')
plt.axhline(T2_limit, color = 'red', linestyle = 'dashed')
plt.scatter([i for i in range(T2_train.shape[0])], T2_train, c='black', s=100, alpha=0.5)
plt.scatter([i for i in range(T2_train.shape[0], T2_train.shape[0]+T2_test.shape[0], 1)], T2_test, c='blue', s=100, alpha=0.5)
plt.show()



# %% Fix for no scatter but only directions
from pca import pca
# Initialize
model = pca()

# Example with DataFrame
X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
X = pd.DataFrame(data=X, columns=np.arange(0, X.shape[1]).astype(str))
# Fit transform
out = model.fit_transform(X)
# Make plot
fig, ax = model.biplot(cmap=None, legend=False, visible=True)
fig, ax = model.biplot(cmap='Set2', legend=False, visible=True)
fig, ax = model.biplot(cmap=None, legend=False, visible=False)


# %%
from pca import pca

X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
outliers = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)
X = np.vstack((X, outliers))

model = pca(alpha=0.05, n_std=2)

# Fit transform
out = model.fit_transform(X)
out['topfeat']
print(out['outliers'])

model.biplot(legend=True, visible=True)

model.biplot(legend=True, SPE=True, HT2=True, visible=True)
model.biplot(legend=True, SPE=True, HT2=False)
model.biplot(legend=True, SPE=False, HT2=True)
model.biplot(legend=True, SPE=False, HT2=False)

model.biplot3d(legend=True, SPE=True, HT2=True)
model.biplot3d(legend=True, SPE=True, HT2=False)
model.biplot3d(legend=True, SPE=False, HT2=True)
model.biplot3d(legend=True, SPE=False, HT2=False)

model.scatter(legend=True, SPE=True, HT2=True)
model.scatter(legend=True, SPE=True, HT2=False)
model.scatter(legend=True, SPE=False, HT2=True)
model.scatter(legend=True, SPE=False, HT2=False)

model.scatter3d(legend=True, SPE=True, HT2=True, visible=True)
model.scatter3d(legend=True, SPE=True, HT2=False)
model.scatter3d(legend=True, SPE=False, HT2=True)
model.scatter3d(legend=True, SPE=False, HT2=False)


ax = model.biplot(n_feat=4, legend=False, )

import pca
outliers_hot = pca.hotellingsT2(out['PC'].values, alpha=0.05)
outliers_spe = pca.spe_dmodx(out['PC'].values, n_std=2)

model.biplot(SPE=True, HT2=False)

# Select the outliers
Xoutliers = X[out['outliers']['y_bool'],:]

# Select the other set
Xnormal = X[~out['outliers']['y_bool'],:]


# %% Transform unseen datapoints into fitted space
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from pca import pca

# Initialize
model = pca(n_components=2, normalize=True, detect_outliers=None)
# Dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)

# Get some random samples across the classes
idx=[0,1,2,3,4,50,51,52,53,54,55,100,101,102,103,104,105]
X_unseen = X.iloc[idx, :]

# Label original dataset to make sure the check which samples are overlapping
X.index.values[idx]=3

# Fit transform
out = model.fit_transform(X)

# Transform new "unseen" data
PCnew = model.transform(X_unseen)

# Plot PC space
model.scatter(alpha=0.5)
# Plot the new "unseen" samples on top of the existing space
plt.scatter(PCnew.iloc[:, 0], PCnew.iloc[:, 1], marker='x')


# %%
# from hnet import hnet
# df = pd.read_csv('C://temp//usarrest.txt')
# hn = hnet(y_min=3, perc_min_num=None)
# results=hn.association_learning(df)
# hn.plot()

# %%

# from pca import pca
# import pandas as pd

# model = pca(normalize=True)
# # Dataset
# df = pd.read_csv('C://temp//usarrest.txt')
# # Setup dataset
# X = df[['Murder','Assault','UrbanPop','Rape']].astype(float)
# X.index = df['state'].values

# # Fit transform
# out = model.fit_transform(X)
# out['topfeat']
# out['outliers']

# ax = model.scatter(legend=False, )
# ax = model.scatter3d(legend=False)

# # Make plot
# ax = model.biplot(n_feat=4, legend=False)
# ax = model.biplot(n_feat=4, legend=False, )

# ax = model.biplot3d(n_feat=1, legend=False)
# ax = model.biplot3d(n_feat=2, legend=False)
# ax = model.biplot3d(n_feat=4, legend=False, )

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

ax = model.biplot(n_feat=10, legend=False, PC=[1, 0])
ax = model.biplot3d(n_feat=10, legend=False)

# %% Normalize out PC1, PC2
X_norm = model.norm(X, pcexclude=[1, 2])
X_norm = pd.DataFrame(data=X_norm, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
out = model.fit_transform(X_norm)
out['topfeat']

model.plot()
ax = model.biplot(n_feat=10, legend=False)


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
# !pip install df2onehot

# Transform data into one-hot
from df2onehot import df2onehot
y = df['Survived'].values
# del df['Survived']
del df['PassengerId']
del df['Name']
out = df2onehot(df, verbose=4)
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

from pca import pca
# Initialize
model2 = pca(normalize=True, onehot=False)
# Run model 2
model2.fit_transform(X)
model2.plot()
model2.biplot(n_feat=4)
model2.scatter()
model2.biplot3d(n_feat=10)

# Set custom transparency levels
model2.biplot3d(n_feat=10, alpha=0.5)
model2.biplot(n_feat=10, alpha=0.5)
model2.scatter3d(alpha=0.5)
model2.scatter(alpha=0.5)


model = pca(normalize=False, method='pca')
model.fit_transform(X)
model.biplot(n_feat=3)

# Initialize
model3 = pca(normalize=False, method='sparse_pca')
# Run model 2
model3.fit_transform(X)
model3.biplot(n_feat=3)

from pca import pca
from scipy.sparse import csr_matrix
S = csr_matrix(X)

model3 = pca(normalize=True, method='trunc_svd')
# Run model 2
model3.fit_transform(X)
model3.biplot(n_feat=3)

# from sklearn.manifold import TSNE
# coord = TSNE(metric='hamming').fit_transform(X)
# from scatterd import scatterd
# scatterd(coord[:, 0], coord[:, 1], labels=y)



#%% Example with Model initialization outside the for-loop.
from pca import pca
model1 = pca(n_components=0.95)
model2 = pca(n_components=0.95)

X = np.array(np.random.normal(0, 1, 5000)).reshape(1000, 5)

for i in range(0, 10):
    I = np.random.randint(1,1000,100)
    model1.fit_transform(X[I,:], verbose=2);
    model2.fit_transform(X[I,:], verbose=2);
    if np.all(model1.results['loadings']==model2.results['loadings']):
        print('Run %d is correct!' %(i))

# %%
