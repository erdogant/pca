Control Arrows
###############


.. code:: python

	# Load pca
	from pca import pca

	# Initialize pca
	model = pca(n_components=3)

	# Load example data set
	df = model.import_example(data='iris')

	# Fit transform
	results = model.fit_transform(df)



.. code:: python

	# Strenght of the loadings are shown in two colors (red=strong, blue=weak).
	model.biplot(s=0)

.. image:: ../figs/wine_arrow_1.png
   :width: 600
   :align: center



.. code:: python

	# Change colors
	model.biplot(s=0, arrowdict={'color_strong': 'r', 'color_weak': 'g'})

.. image:: ../figs/wine_arrow_2.png
   :width: 600
   :align: center


.. code:: python

	# The transparancy of the arrows are set on the loading value. You can also set it to a constant value.
	model.biplot(s=0, arrowdict={'alpha': 0.8})

.. image:: ../figs/wine_arrow_3.png
   :width: 600
   :align: center


.. code:: python

	# Change text color
	model.biplot(s=0, arrowdict={'color_text': 'k'})

.. image:: ../figs/wine_arrow_4.png
   :width: 600
   :align: center


.. code:: python

	# Change arrow color.
	model.biplot(s=0, color_arrow='k')

.. image:: ../figs/wine_arrow_5.png
   :width: 600
   :align: center



.. code:: python

	# Set color arrow and color text
	model.biplot(s=0, color_arrow='k', arrowdict={'color_text': 'g'})

.. image:: ../figs/wine_arrow_8.png
   :width: 600
   :align: center



.. code:: python

	# Default settings
	model.biplot()

.. image:: ../figs/wine_arrow_9.png
   :width: 600
   :align: center


.. code:: python

	# Change the scale factor of the arrow. The scale_factor of 3 seems a little bit too much here ;)
	model.biplot3d(arrowdict={'scale_factor': 3})


.. image:: ../figs/wine_arrow_10.png
   :width: 600
   :align: center


.. include:: add_bottom.add