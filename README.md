# pca

[![Python](https://img.shields.io/pypi/pyversions/pca)](https://img.shields.io/pypi/pyversions/pca)
[![PyPI Version](https://img.shields.io/pypi/v/pca)](https://pypi.org/project/pca/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/pca/blob/master/LICENSE)

* pca is a python package that performs the principal component analysis and allows to make several plots; biplot to plot the loadings of the PCA, explained variance plot and a simple scatter plot.

## Method overview


## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install pca from PyPI (recommended). pca is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

## Requirements
* It is advisable to create a new environment. 
```python
conda create -n env_pca python=3.6
conda activate env_pca
pip install numpy pandas tqdm matplotlib
```

## Quick Start
```
pip install pca
```

* Alternatively, install pca from the GitHub source:
```bash
git clone https://github.com/erdogant/pca.git
cd pca
python setup.py install
```  

## Import pca package
```python
import pca as pca
```

## Example: Structure Learning
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/pca/data/example_data.csv')
model = pca.structure_learning(df)
G = pca.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig1.png" width="600" />
  
</p>

* Choosing various methodtypes and scoringtypes:
```python
model_hc_bic  = pca.structure_learning(df, methodtype='hc', scoretype='bic')
```

#### df looks like this:
```
     Cloudy  Sprinkler  Rain  Wet_Grass
0         0          1     0          1
1         1          1     1          1
2         1          0     1          1
3         0          0     1          1
4         1          0     1          1
..      ...        ...   ...        ...
995       0          0     0          0
996       1          0     0          0
997       0          0     1          0
998       1          1     0          1
999       1          0     1          1
```


## Citation
Please cite pca in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019pca,
  title={pca},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/pca}},
}
```

## References
* http://pgmpy.org
* https://programtalk.com/python-examples/pgmpy.factors.discrete.TabularCPD/
* http://www.pca.com/
* http://www.pca.com/bnrepository/
   
## Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

## Contribute
* Contributions are welcome.

## © Copyright
See [LICENSE](LICENSE) for details.
