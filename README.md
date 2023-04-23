
<p align="center">
  <a href="https://erdogant.github.io/pca/">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/iris_density.png" width="600" />
  </a>
</p>



[![Python](https://img.shields.io/pypi/pyversions/pca)](https://img.shields.io/pypi/pyversions/pca)
[![Pypi](https://img.shields.io/pypi/v/pca)](https://pypi.org/project/pca/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/pca/)
[![LOC](https://sloc.xyz/github/erdogant/pca/?category=code)](https://github.com/erdogant/pca/)
[![Downloads](https://static.pepy.tech/personalized-badge/pca?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/pca)
[![Downloads](https://static.pepy.tech/personalized-badge/pca?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/pca)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/pca/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/pca.svg)](https://github.com/erdogant/pca/network)
[![Open Issues](https://img.shields.io/github/issues/erdogant/pca.svg)](https://github.com/erdogant/pca/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/233232884.svg)](https://zenodo.org/badge/latestdoi/233232884)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/pca/pages/html/Documentation.html#colab-notebook)
![GitHub repo size](https://img.shields.io/github/repo-size/erdogant/pca)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/pca/pages/html/Documentation.html#)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

**pca** A Python Package for Principal Component Analysis. The core of PCA is build on sklearn functionality to find maximum compatibility when combining with other packages.
But this package can do a lot more. Besides the regular pca, it can also perform **SparsePCA**, and **TruncatedSVD**. Depending on your input data, the best approach will be choosen.

Other functionalities of PCA are:

  * **Biplot** to plot the loadings
  * Determine the **explained variance** 
  * Extract the best performing **features**
  * Scatter plot with the **loadings**
  * Outlier detection using **Hotelling T2 and/or SPE/Dmodx**

---

**⭐️ Star this repo if you like it ⭐️**

---

## Read the Medium blog for more details.

<p align="left">
  <a href="https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/medium_blog1.png?raw=true" width="150" />
  </a>
</p>


---

## [Documentation pages](https://erdogant.github.io/pca/)

On the [documentation pages](https://erdogant.github.io/pca/) you can find detailed information about the working of the ``pca`` with many examples. 

---

## Installation

```bash
pip install pca
```

##### Import pca package

```python
from pca import pca
```

# 

<table style="width:100%">
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Examples.html">Quick start</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot">Make biplot</a></th>
  </tr>
  <tr>
    <td>
      <p align="left">
        <a href="https://erdogant.github.io/pca/pages/html/Examples.html">
          <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter.png?raw=true" width="400" />
        </a>
      </p>    
    </td>
    <td>
      <p align="left">
        <a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot">
          <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_biplot.png?raw=true" width="350" />
        </a>
      </p>
    </td>
  </tr>
</table>



# 

<table style="width:100%">
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#explained-variance-plot">Plot Explained variance</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#d-plots">3D plots</a></th>
  </tr>
  <tr>
    <td>
      <p align="left">
        <a href="https://erdogant.github.io/pca/pages/html/Plots.html#explained-variance-plot">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_plot.png" width="350" />
        </a>
      </p>
    </td>
    <td>
      <p align="left">
        <a href="https://erdogant.github.io/pca/pages/html/Plots.html#d-plots">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter3d.png" width="350" />
        </a>
      </p>
    </td>
  </tr>
</table>



# 

* [Example: Set alpha transparency](https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency)

<p align="left">
  <a href="https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter.png" width="350" />
  </a>
</p>


# 

* [Example: Normalizing out Principal Components](https://erdogant.github.io/pca/pages/html/Algorithm.html#normalizing-out-pcs)

Normalizing out the 1st and more components from the data. 
This is usefull if the data is seperated in its first component(s) by unwanted or biased variance. Such as sex or experiment location etc. 

# 

* [Example: Extract Feature Importance](https://erdogant.github.io/pca/pages/html/Examples.html#feature-importance)

Make the biplot. It can be nicely seen that the first feature with most variance (f1), is almost horizontal in the plot, whereas the second most variance (f2) is almost vertical. This is expected because most of the variance is in f1, followed by f2 etc.

[![Explained variance][1]][1]


Biplot in 2d and 3d. Here we see the nice addition of the expected f3 in the plot in the z-direction.

[![biplot][2]][2]

[![biplot3d][3]][3]


  [1]: https://i.stack.imgur.com/Wb1rN.png
  [2]: https://i.stack.imgur.com/V6BYZ.png
  [3]: https://i.stack.imgur.com/831NF.png
  

# 

* [Example: Detection of outliers](https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency)

To detect any outliers across the multi-dimensional space of PCA, the *hotellings T2* test is incorporated. 
This basically means that we compute the chi-square tests across the top n_components (default is PC1 to PC5).
It is expected that the highest variance (and thus the outliers) will be seen in the first few components because of the nature of PCA.
Going deeper into PC space may therefore not required but the depth is optional.
This approach results in a P-value matrix (samples x PCs) for which the P-values per sample are then combined using *fishers* method. 
This approach allows to determine outliers and the ranking of the outliers (strongest tot weak). The alpha parameter determines the detection of outliers (default: 0.05).

<p align="left">
  <a href="https://erdogant.github.io/pca/pages/html/Outlier%20detection.html">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/outliers_biplot_spe_hot.png" width="350" />
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/outliers_biplot3d.png" width="350" />
  </a>
</p>



# 

* [Example: Plot only the loadings (arrows)](https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency)

<p align="left">
  <a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot-only-arrows">
  <img src="https://github.com/erdogant/pca/blob/master/docs/figs/biplot_only_directions.png" width="350" />
  </a>
</p>


# 

* [Example: Selection of outliers](https://erdogant.github.io/pca/pages/html/Outlier%20detection.html#selection-of-the-outliers)

# 

* [Example: Toggle visible status](https://erdogant.github.io/pca/pages/html/Plots.html#toggle-visible-status)

# 

* [Example: Map unseen (new) datapoint to the transfomred space](https://erdogant.github.io/pca/pages/html/Examples.html#map-unseen-datapoints-into-fitted-space)

<hr>

#### Citation
Please cite in your publications if this is useful for your research (see citation).

### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

---

## Support

Your ❤️ is important to keep maintaining this package. You can [support](https://erdogant.github.io/pca/pages/html/Documentation.html) in various ways, have a look at the [sponser page](https://erdogant.github.io/pca/pages/html/Documentation.html). Report bugs, issues and feature extensions at [github page](https://github.com/erdogant/pca).

<a href='https://www.buymeacoffee.com/erdogant' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://cdn.ko-fi.com/cdn/kofi1.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
