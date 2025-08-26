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
[![Medium](https://img.shields.io/badge/Medium-Blog-purple)](https://erdogant.medium.com)
[![Gumroad](https://img.shields.io/badge/Gumroad-Blog-purple)](https://erdogant.gumroad.com/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/pca/pages/html/Documentation.html#colab-notebook)
![GitHub repo size](https://img.shields.io/github/repo-size/erdogant/pca)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/pca/pages/html/Documentation.html#)

### 

<div>

<a href="https://erdogant.github.io/pca/"><img src="https://github.com/erdogant/pca/blob/master/docs/figs/iris_density.png" width="175" align="left" /></a>
``pca`` is a Python package for Principal Component Analysis. The core of PCA is built on sklearn functionality to find maximum compatibility when combining with other packages.
But this package can do a lot more. Besides the regular PCA, it can also perform SparsePCA, and TruncatedSVD. Depending on your input data, the best approach can be chosen.
``pca`` contains the most-wanted analysis and plots. Navigate to [API documentations](https://erdogant.github.io/pca/) for more detailed information. **⭐️ Star it if you like it ⭐️**
</div>

---


### Key Features

| Feature | Description |
|--------|-------------|
| [**Fit and Transform**](https://erdogant.github.io/pca/pages/html/Algorithm.html) | Perform the PCA analysis. |
| [**Biplot and Loadings**](https://erdogant.github.io/pca/pages/html/Plots.html#biplot) | Make Biplot with the loadings. |
| [**Explained Variance**](https://erdogant.github.io/pca/pages/html/Plots.html#explained-variance-plot) | Determine the explained variance and plot. |
| [**Best Performing Features**](https://erdogant.github.io/pca/pages/html/Algorithm.html#best-performing-features) | Extract the best performing features. |
| [**Scatterplot**](https://erdogant.github.io/pca/pages/html/Plots.html#scatter-plot) | Create scaterplot with loadings. |
| [**Outlier Detection**](https://erdogant.github.io/pca/pages/html/Outlier%20detection.html) | Detect outliers using Hotelling T2 and/or SPE/Dmodx. |
| [**Normalize out Variance**](https://erdogant.github.io/pca/pages/html/Examples.html#normalizing-out-pcs) | Remove any bias from your data. |
| [**Save and load**](https://erdogant.github.io/pca/pages/html/save.html) | Save and load models. |
| [**Analyze discrete datasets**](https://erdogant.github.io/pca/pages/html/Examples.html#analyzing-discrete-datasets) | Analyze discrete datasets. |

---

### Resources and Links
- **Example Notebooks:** [Examples](https://erdogant.github.io/pca/pages/html/Documentation.html#colab-notebook)
- **Blog: PCA and Loadings:** [Medium](https://medium.com/data-science-collective/pca-fb6ea1208bda)
- **Blog: Outlier Detection:** [Medium](https://medium.com/data-science-collective/outlier-detection-using-principal-component-analysis-with-hotellings-t2-and-spe-dmodx-methods-c9c0c76cc6c7)
- **Blog and podcast:** [GumRoad](https://erdogant.gumroad.com/l/PCA)
- **Documentation:** [Website](https://erdogant.github.io/pca)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/erdogant/pca/issues)

---


## Installation

```bash
pip install pca
```


```python
from pca import pca
```

---
## Examples

<table style="width:100%">

  <!-- Row 1 -->
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Examples.html">Quick Start</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot">Make Biplot</a></th>
  </tr>
  <tr>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Examples.html">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter.png?raw=true" width="400" />
      </a>
    </td>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/custom_example_2.png?raw=true" width="350" />
      </a>
    </td>
  </tr>

  <!-- Row 2 -->
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#explained-variance-plot">Explained Variance Plot</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#d-plots">3D Plots</a></th>
  </tr>
  <tr>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Plots.html#explained-variance-plot">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_plot.png" width="350" />
      </a>
    </td>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Plots.html#d-plots">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/iris_3d_density.png" width="350" />
      </a>
    </td>
  </tr>

  <!-- Row 3 -->
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency">Alpha Transparency</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Algorithm.html#normalizing-out-pcs">Normalize Out Principal Components</a></th>
  </tr>
  <tr>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Plots.html#alpha-transparency">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/fig_scatter.png" width="350" />
      </a>
    </td>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Algorithm.html#normalizing-out-pcs">
        <img src="https://i.stack.imgur.com/Wb1rN.png" width="350" />
      </a>
    </td>
  </tr>

  <!-- Row 4: Feature Importance -->
  <tr>
    <th colspan="2"><a href="https://erdogant.github.io/pca/pages/html/Examples.html#feature-importance">Extract Feature Importance</a></th>
  </tr>
  <tr>
    <td colspan="2">
      Make the biplot to visualize the contribution of each feature to the principal components.
      <br/><br/>
      <a href="https://i.stack.imgur.com/V6BYZ.png">
        <img src="https://i.stack.imgur.com/V6BYZ.png" width="350" />
      </a>
      <a href="https://i.stack.imgur.com/831NF.png">
        <img src="https://i.stack.imgur.com/831NF.png" width="350" />
      </a>
    </td>
  </tr>

  <!-- Row 5 -->
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Outlier%20detection.html">Detect Outliers</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot-only-arrows">Show Only Loadings</a></th>
  </tr>
  <tr>
    <td align="left">
      Detect outliers using Hotelling's T² and Fisher’s method across top components (PC1–PC5).
      <br/><br/>
      <a href="https://erdogant.github.io/pca/pages/html/Outlier%20detection.html">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/outliers_biplot_spe_hot.png" width="170" />
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/outliers_biplot3d.png" width="170" />
      </a>
    </td>
    <td align="left">
      <a href="https://erdogant.github.io/pca/pages/html/Plots.html#biplot-only-arrows">
        <img src="https://github.com/erdogant/pca/blob/master/docs/figs/biplot_only_directions.png" width="350" />
      </a>
    </td>
  </tr>

  <!-- Row 6 -->
  <tr>
    <th><a href="https://erdogant.github.io/pca/pages/html/Outlier%20detection.html#selection-of-the-outliers">Select Outliers</a></th>
    <th><a href="https://erdogant.github.io/pca/pages/html/Plots.html#toggle-visible-status">Toggle Visibility</a></th>
  </tr>
  <tr>
    <td align="left">
      Select and filter identified outliers for deeper inspection or removal.
    </td>
    <td align="left">
      Toggle visibility of samples and components to clean up visualizations.
    </td>
  </tr>

  <!-- Row 7 -->
  <tr>
    <th colspan="2"><a href="https://erdogant.github.io/pca/pages/html/Examples.html#map-unseen-datapoints-into-fitted-space">Map Unseen Datapoints</a></th>
  </tr>
  <tr>
    <td colspan="2">
      Project new data into the transformed PCA space. This enables testing new observations without re-fitting the model.
    </td>
  </tr>

</table>

---

### Contributors
Setting up and maintaining PCA has been possible thanks to users and contributors. Thanks to:

<p align="left">
  <a href="https://github.com/erdogant/pca/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erdogant/pca" />
  </a>
</p>

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
