# pca

[![Python](https://img.shields.io/pypi/pyversions/pca)](https://img.shields.io/pypi/pyversions/pca)
[![PyPI Version](https://img.shields.io/pypi/v/pca)](https://pypi.org/project/pca/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/pca/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/pca)](https://pepy.tech/project/pca)

         Star it if you like it!

* pca is a python package that performs the principal component analysis and creates insightful plots.
* Biplot to plot the loadings
* Explained variance 
* Scatter plot with the loadings

## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Contribute](#-contribute)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install pca from PyPI (recommended). pca is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

### Requirements
* Creation of a new environment is not required but if you wish to do it:
```python
conda create -n env_pca python=3.6
conda activate env_pca
pip install numpy matplotlib sklearn
```

### Installation
```
pip install pca
```

* Install the latest version from the GitHub source:
```bash
git clone https://github.com/erdogant/pca.git
cd pca
python setup.py install
```  

#### Import pca package
```python
from pca import pca
```

#### Load example data
```python
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
```

#### X looks like this:
```
X=array([[5.1, 3.5, 1.4, 0.2],
         [4.9, 3. , 1.4, 0.2],
         [4.7, 3.2, 1.3, 0.2],
         [4.6, 3.1, 1.5, 0.2],
         ...
         [5. , 3.6, 1.4, 0.2],
         [5.4, 3.9, 1.7, 0.4],
         [4.6, 3.4, 1.4, 0.3],
         [5. , 3.4, 1.5, 0.2],

labx=[0, 0, 0, 0,...,2, 2, 2, 2, 2]
label=['label1','label2','label3','label4']
```


# Make scatter plot
```python
fig, ax = model.scatter()
```
<p align="center">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter.png" width="400" />
</p>


# Make biplot
```python
fig, ax = model.biplot(n_feat=4)
```
<p align="center">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_biplot.png" width="350" />
</p>

# Make plot
```python
fig, ax = model.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_plot.png" width="350" />
</p>

# Make 3d plots
```python
fig, ax model.scatter3d()
fig, ax = model.biplot3d(n_feat=2)
```
<p align="center">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter3d.png" width="350" />
</p>


#### PCA normalization. 
Normalizing out the 1st and more components from the data. 
This is usefull if the data is seperated in its first component(s) by unwanted or biased variance. Such as sex or experiment location etc. 

```python
print(X.shape)
(150, 4)

# Normalize out 1st component and return data
model = pca()
Xnew = model.norm(X, pcexclude=[1])


print(Xnorm.shape)
(150, 4)

# In this case, PC1 is "removed" and the PC2 has become PC1 etc
ax = pca.biplot(model)

```



the pca library contains this functionality.

    pip install pca

A demonstration to extract the feature importance is as following:

    # Import libraries
    import numpy as np
    import pandas as pd
    from pca import pca

    # Lets create a dataset with features that have decreasing variance. 
    # We want to extract feature f1 as most important, followed by f2 etc
    f1=np.random.randint(0,100,250)
    f2=np.random.randint(0,50,250)
    f3=np.random.randint(0,25,250)
    f4=np.random.randint(0,10,250)
    f5=np.random.randint(0,5,250)
    f6=np.random.randint(0,4,250)
    f7=np.random.randint(0,3,250)
    f8=np.random.randint(0,2,250)
    f9=np.random.randint(0,1,250)

    # Combine into dataframe
    X = np.c_[f1,f2,f3,f4,f5,f6,f7,f8,f9]
    X = pd.DataFrame(data=X, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
    
    # Initialize
    model = pca()
    # Fit transform
    out = model.fit_transform(X)

    # Print the top features. The results show that f1 is best, followed by f2 etc
    print(out['topfeat'])

    #     PC      feature
    # 0  PC1      f1
    # 1  PC2      f2
    # 2  PC3      f3
    # 3  PC4      f4
    # 4  PC5      f5
    # 5  PC6      f6
    # 6  PC7      f7
    # 7  PC8      f8
    # 8  PC9      f9

Plot the explained variance

    model.plot()

[![Explained variance][1]][1]

Make the biplot. It can be nicely seen that the first feature with most variance (f1), is almost horizontal in the plot, whereas the second most variance (f2) is almost vertical. This is expected because most of the variance is in f1, followed by f2 etc.

    ax = model.biplot(n_feat=10, legend=False)

[![biplot][2]][2]

Biplot in 3d. Here we see the nice addition of the expected f3 in the plot in the z-direction.

    ax = model.biplot3d(n_feat=10, legend=False)

[![biplot3d][3]][3]


  [1]: https://i.stack.imgur.com/Wb1rN.png
  [2]: https://i.stack.imgur.com/V6BYZ.png
  [3]: https://i.stack.imgur.com/831NF.png
  
### Citation
Please cite distfit in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019pca,
  title={pca},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/pca}},
}
```


### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
