Background
###########

``pca`` is a python package to perform *Principal Component Analysis* and to examine the variance in-depth. The core of PCA is build on **sklearn** to find maximum compatibility when combining with other packages. But this package can do a lot more. Besides the regular **Principal Components**, it also integrates **SparsePCA**, **TruncatedSVD**, and provides the information that can be extracted from the components. 

Functionalities of PCA are:

	* Biplot to plot the loadings
	* Determine the explained variance
	* Extract the best performing features
	* Scatter plot with the loadings
	* Outlier detection using Hotelling T2 and/or SPE/Dmodx
	* Removing unwantend (technical) bias from the data

    


.. include:: add_bottom.add