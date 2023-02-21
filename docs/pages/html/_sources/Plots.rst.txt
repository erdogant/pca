Load dataset
##############################

Let's load the **wine** dataset to demonstrate the plots.

.. code:: python

	# Load library
	from sklearn.datasets import load_wine
	# Load dataset
	data = load_wine()
	X = data.data
	y = data.target
	labels = data.feature_names

	from pca import pca
	# Initialize
	model = pca(normalize=True)

	# Fit transform and include the column labels and row labels
	results = model.fit_transform(X, col_labels=labels, row_labels=y)

	# [pca] >Normalizing input data per feature (zero mean and unit variance)..
	# [pca] >The PCA reduction is performed to capture [95.0%] explained variance using the [13] columns of the input data.
	# [pca] >Fitting using PCA..
	# [pca] >Computing loadings and PCs..
	# [pca] >Computing explained variance..
	# [pca] >Number of components is [10] that covers the [95.00%] explained variance.
	# [pca] >The PCA reduction is performed on the [13] columns of the input dataframe.
	# [pca] >Fitting using PCA..
	# [pca] >Computing loadings and PCs..
	# [pca] >Outlier detection using Hotelling T2 test with alpha=[0.05] and n_components=[10]
	# [pca] >Outlier detection using SPE/DmodX with n_std=[2]


Scatter plot
###############


.. code:: python

	# Make scatterplot
	model.scatter()

	# Gradient over the samples. High dense areas will be more colourful.
	model.scatter(gradient='#FFFFFF')

	# Include the outlier detection
	model.scatter(SPE=True)

	# Include the outlier detection
	model.scatter(hotellingt2=True)

	# Look at different PCs: 1st PC=1  vs PC=3
	model.scatter(PC=[0, 2])


.. |figP1| image:: ../figs/wine_scatter.png
.. |figP2| image:: ../figs/wine_scatter_spe.png
.. |figP3| image:: ../figs/wine_scatter_hotel.png
.. |figP4| image:: ../figs/wine_scatter_PC13.png
.. |figP7| image:: ../figs/wine_scatter_density.png

.. table:: Scatterplots
   :align: center

   +----------+
   | |figP1|  |
   +----------+
   | |figP7|  |
   +----------+
   | |figP2|  |
   +----------+
   | |figP3|  |
   +----------+
   | |figP4|  |
   +----------+


Biplot
###############

.. code:: python

	# Make biplot
	model.biplot()

	# Here again, many other options can be turned on and off
	model.biplot(SPE=True, hotellingt2=True)


.. |figP5| image:: ../figs/wine_biplot.png
.. |figP6| image:: ../figs/wine_biplot_with_outliers.png

.. table:: Biplots
   :align: center

   +----------+
   | |figP5|  |
   +----------+
   | |figP6|  |
   +----------+

\

Biplot (only arrows)
########################

.. code:: python

	# Make plot with parameters: set cmap to None and label and legend to False. Only directions will be plotted.
	model.biplot(cmap=None, label=False, legend=False)


.. image:: ../figs/biplot_only_directions.png
   :width: 600
   :align: center


Explained variance plot
##############################


.. code:: python

	model.plot()

.. image:: ../figs/wine_explained_variance.png
   :width: 600
   :align: center



Alpha Transparency
##############################

.. code:: python

	fig, ax = model.scatter(alpha_transparency=1)


3D plots
###############

All plots can also be created in 3D by setting the ``d3=True`` parameter.

.. code:: python

	model.biplot3d()


Toggle visible status
##############################

The visible status for can be turned on and off.

.. code:: python

	# Make plot but not visible.
	fig, ax = model.biplot(visible=False)

	# Set the figure again to True and show the figure.
	fig.set_visible(True)
	fig



.. include:: add_bottom.add