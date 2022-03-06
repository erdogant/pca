Quickstart
############

A quick example how perform feature reduction using ``pca`` on a given dataset.

.. code:: python

	import numpy as np
	from sklearn.datasets import load_iris

	# Load dataset
	label = load_iris().feature_names
	y = load_iris().target
	X = pd.DataFrame(data=load_iris().data, columns=label, index=y)

	# Load pca
	from pca import pca

	# Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
	model = pca(n_components=0.95)

	# Reduce the data towards 3 PCs
	model = pca(n_components=3)

	# Fit transform
	results = model.fit_transform(X)

	# Data looks like this:

	# X=array([[5.1, 3.5, 1.4, 0.2],
	# 	 [4.9, 3. , 1.4, 0.2],
	# 	 [4.7, 3.2, 1.3, 0.2],
	# 	 [4.6, 3.1, 1.5, 0.2],
	# 	 ...
	# 	 [5. , 3.6, 1.4, 0.2],
	# 	 [5.4, 3.9, 1.7, 0.4],
	# 	 [4.6, 3.4, 1.4, 0.3],
	# 	 [5. , 3.4, 1.5, 0.2],
	# 
	# y = [0, 0, 0, 0,...,2, 2, 2, 2, 2]
	# label = ['sepal length (cm)',
	#	 'sepal width (cm)',
	#	 'petal length (cm)',
	#	 'petal width (cm)']


Explained variance plot
************************************

.. code:: python

	fig, ax = model.plot()

.. image:: ../figs/fig_plot.png
   :width: 600
   :align: center


Scatter plot
******************

.. code:: python

	# 2D plot
	fig, ax = model.scatter()

	# 3d Plot
	fig, ax = model.scatter3d()


.. |figE1| image:: ../figs/fig_scatter.png
.. |figE2| image:: ../figs/fig_scatter3d.png

.. table:: Color on alcohol
   :align: center

   +----------+----------+
   | |figE1|  | |figE2|  |
   +----------+----------+


Biplot
******************

.. code:: python
	
	# 2D plot
	fig, ax = model.biplot(n_feat=4, PC=[0,1])

	# 3d Plot
	fig, ax = model.biplot3d(n_feat=2, PC=[0,1,2])

.. image:: ../figs/fig_biplot.png
   :width: 600
   :align: center




Feature importance
#####################################################

This example is created to showcase the working of extracting features that are most important in a PCA reduction.
We will create random variables with increasingly more variance. The first feature (f1) will have most of the variance, followed by feature 2 (f2) etc.


.. code:: python

	# Print the top features.
	print(model.results['topfeat'])

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

	# Initialize and keep all PCs
	model = pca()
	# Fit transform
	out = model.fit_transform(X)

	# Print the top features.
	print(out['topfeat'])

	# The results show the expected results: f1 is the best, followed by f2 etc
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


Explained variance plot
****************************

.. code:: python

	model.plot()

.. image:: ../figs/explained_var_1.png
   :width: 600
   :align: center



Biplot
****************************

Make the biplot. It can be nicely seen that the first feature with most variance (f1), is almost horizontal in the plot, whereas the second most variance (f2) is almost vertical. This is expected because most of the variance is in f1, followed by f2 etc. Biplot in 3d. Here we see the nice addition of the expected f3 in the plot in the z-direction.


.. code:: python

	# 2d plot
	ax = model.biplot(n_feat=10, legend=False)

	# 3d plot
	ax = model.biplot3d(n_feat=10, legend=False)



.. |figA1| image:: ../figs/biplot2d.png
.. |figA2| image:: ../figs/biplot3d.png

.. table:: Color on alcohol
   :align: center

   +----------+----------+
   | |figA1|  | |figA2|  |
   +----------+----------+




Analyzing Discrete datasets
#####################################################

Analyzing datasets that have continuous and catagorical values can be challanging.
To demonstrate how to do this, I will use the Titanic dataset. We need to pip install df2onehot first.

.. code:: bash

	pip install df2onehot


.. code:: python

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


Map unseen datapoints into fitted space
##############################################

After fitting variables into the new principal component space, we can map new unseen samples into this space too. However, there is also normalization step which can be tricky because you now need standardize the values of the unseen samples first based on the previously performed standardization. This step is also integrated in the ``pca`` library by simply setting the parameter ``normalize=True``.


.. code:: python

	# Load libraries
	import matplotlib.pyplot as plt
	from sklearn import datasets
	import pandas as pd
	from pca import pca

	# Load dataset
	data = datasets.load_wine()
	X = data.data
	y = data.target.astype(str)
	labels = data.feature_names

	# Initialize with normalization and take the number of components that covers at least 95% of the variance.
	model = pca(n_components=0.95, normalize=True)


	# Get some random samples across the classes
	idx=[0,1,2,3,4,50,53,54,55,100,103,104,105, 130, 150]
	X_unseen = X[idx, :]
	y_unseen = y[idx]

	# Label original dataset to make sure the check which samples are overlapping
	y[idx]='unseen'

	# Fit transform
	model.fit_transform(X, col_labels=col_labels, row_labels=y)

	# Transform new "unseen" data. Note that these datapoints are not really unseen as they are readily fitted above.
	# But for the sake of example, you can see that these samples will be transformed exactly on top of the orignial ones.
	PCnew = model.transform(X_unseen)

	# Plot PC space
	model.scatter(title='Map unseen samples in the existing space.')
	# Plot the new "unseen" samples on top of the existing space
	plt.scatter(PCnew.iloc[:, 0], PCnew.iloc[:, 1], marker='x', s=200)


.. image:: ../figs/wine_mapping_samples.png
   :width: 600
   :align: center


Normalizing out PCs
#########################

Normalize your data using the principal components. As an example, suppose there is (technical) variation in the fist component and you want that out. This function transforms the data using the components that you want, e.g., starting from the 2nd PC, up to the OC that contains at least 95% of the explained variance.


.. code:: python

	print(X.shape)
	(178, 13)

	# Normalize out 1st component and return data
	Xnorm = model.norm(X, pcexclude=[1])

	# The data remains the same samples and variables but the all variance that covered the 1st PC is removed.
	print(Xnorm.shape)
	(178, 13)

	# In this case, PC1 is "removed" and the PC2 has become PC1 etc
	ax = pca.biplot(model, col_labels=col_labels, row_labels=y)




.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
