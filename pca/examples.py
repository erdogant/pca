from pca import pca
import pandas as pd
import numpy as np

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
model.scatter(alpha_transparency=0.5)
# Plot the new "unseen" samples on top of the existing space
plt.scatter(PCnew.iloc[:, 0], PCnew.iloc[:, 1], marker='x')


# %% Fix for no scatter but only directions
from pca import pca
# Initialize
model = pca()

# Example with DataFrame
X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
X = pd.DataFrame(data=X, columns=np.arange(0, X.shape[1]).astype(str))
# Fit transform
out = model.fit_transform(X)
out = model.fit_transform(X)
# Make plot
fig, ax = model.biplot(cmap=None, label=False, legend=False, visible=True)
fig, ax = model.biplot(cmap=None, label=False, legend=False, visible=False)


# %%
from pca import pca

# Initialize
model = pca(alpha=0.05, n_std=2)

# Example with Numpy array
X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
row_labels = np.arange(0, X.shape[0]).astype(str)
# Fit transform
out = model.fit_transform(X, row_labels=row_labels)
# Make plot
model.biplot(legend=False)

# Example with DataFrame
X = pd.DataFrame(data=X, columns=np.arange(0, X.shape[1]).astype(str))
# Fit transform
out = model.fit_transform(X)
# Make plot
model.biplot(legend=False)


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

model.biplot(legend=True, SPE=True, hotellingt2=True, visible=True)
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

model.scatter3d(legend=True, SPE=True, hotellingt2=True, visible=True)
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
model2.biplot3d(n_feat=10, alpha_transparency=0.5)
model2.biplot(n_feat=10, alpha_transparency=0.5)
model2.scatter3d(alpha_transparency=0.5)
model2.scatter(alpha_transparency=0.5)

# Initialize
model3 = pca(normalize=False, onehot=True)
# Run model 2
_=model3.fit_transform(X)
model3.biplot(n_feat=3)

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
