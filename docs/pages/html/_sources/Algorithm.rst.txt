Algorithm
#################

Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.


Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for *simplicity*. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

The idea of PCA is simple — **reduce the number of variables** of a data set, while **preserving** as much **information** as possible.

The ``pca`` library contains various functionalities to carefully examine the data, which can help for better understanding and removal of redundant information.


Standardization
*********************

Feature scaling through standardization (or Z-score normalization) is an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one. The standardization step depends on the type of variables, the distribution of the data, and your aim. In general, the standardizing step is to range the continuous initial variables so that each one of them contributes *equally* to the analysis.

It is utterly important to carefully standardize your data because PCA works under the assumption that the data is *normal distributed*, and is very *sensitive* to the variances of the variables. Or in other words, large differences between the ranges of variables will dominate over those with small ranges. Let me explain this by example; a variable that ranges between 0 and 100 will dominate over a variable that ranges between 0 and 1. Transforming the data to comparable scales can prevent this problem.

The most straightforward manner is by computing the *Z-scores* or *Standardized scores*.
Once the standardization is done, all the variables will be transformed to the same scale.

.. |figA7| image:: ../figs/z_score.svg

.. table::
   :width: 80
   :align: left

   +----------+
   | |figA7|  |
   +----------+



Scaling your data can easily being done with the sklearn library. In the following example we will import the well known **wine dataset** and scale the variables. Think carefully whether you want to standardize column-wise or row-wise. In general, you want to standardize row-wise. This means that the Z-score is computer per row.

.. code:: python

	# Import the StandardScaler
	from sklearn.preprocessing import StandardScaler
	from sklearn import datasets

	# Load dataset
	data = datasets.load_wine()
	X = data.data
	y = data.target
	labels = data.feature_names

	#In general it is a good idea to scale the data
	scaler = StandardScaler(with_mean=True, with_std=True)
	X = scaler.fit_transform(X)


The normalization step is also incorporated in ``pca`` that can be set by the parameter ``normalize=True``.

.. code:: python

	# Load library
	from pca import pca

	# Initialize pca with default parameters
	model = pca(normalize=True)


An example of the differences of feature reduction using PCA with and without standardization.

.. |figA3| image:: ../figs/wine_no_standardization.png
.. |figA4| image:: ../figs/wine_yes_standardization.png

.. table:: Without standardization (left) and with standardization (right)
   :align: center

   +----------+----------+
   | |figA3|  | |figA4|  |
   +----------+----------+





Explained Variance
*********************

Before getting to the explanation of **explained variance**, we first need to understand what principal components are.

**Principal components** are new **variables** that are constructed as **linear combinations** or **mixtures** of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. 
**Explained variance** refers to the variance explained by each of the principal components (eigenvectors). By organizing information Principal 

Let's compute the explained variance for the wine dataset (this is a follow up from the previous standardization part).

.. code:: python
	
	# Load library
	from pca import pca

	# Initialize pca with default parameters
	model = pca(normalize=True)

	# Fit transform
	results = model.fit_transform(X)

	# Plot the explained variance
	model.plot()


In this example we have 13 variables in the **wine** dataset, and thus 13 dimensions. PCA will try to put maximum possible information in the first component, then maximum remaining information in the second and so on, until having something like shown in the plot below. This plot will help us to retrieve the insights in the amount of *information* or *explained variance* in the data. We can clearly see that the 1st PC contains almost 36% of explained variance in total. With the top 10 PCs we cover 97.9% of all variance.


.. image:: ../figs/wine_explained_variance.png
   :width: 600
   :align: center


There are as many principal components as there are variables in the data. The **explained variance plot** can therefore never have more then 13 PCs in this case. Principal components are constructed in such a manner that the first principal component accounts for the largest possible variance in the data set.

Loadings
**************************

An important thing to realize here is that, the principal components are less interpretable and don’t have any real meaning since they are constructed as **linear combinations** of the initial variables. But we can analyze the **loadings** which describe the importance of the independent variables.
The first principal component (Y1) is given by a linear combination of the variables X1, X2, ..., Xp, and is calculated such that it accounts for the greatest possible variance in the data. 

.. image:: ../figs/PCAequation1.png
   :width: 300

Of course, one could make the variance of Y1 as large as possible by choosing large values for the weights a11, a12, ... a1p. To prevent this, the sum of squares of the weights is constrained to be 1.


.. image:: ../figs/PCAequation3.png
   :width: 300

For example, let’s assume that the scatter plot of our data set is as shown below, can we guess the first principal component ? Yes, it’s approximately the line that matches the purple marks because it goes through the origin and it’s the line in which the projection of the points (red dots) is the most spread out. Or mathematically speaking, it’s the line that maximizes the variance (the average of the squared distances from the projected points (red dots) to the origin).

.. image:: ../figs/PCA_rotation.gif
   :width: 900
   :align: center


The second principal component is calculated in the same way, with the conditions that it is uncorrelated with (i.e., perpendicular to) the first principal component and that it accounts for the next highest variance.

.. image:: ../figs/PCAequation4.png
   :width: 300


This continues until a total of p principal components have been calculated, that is, the number of principal components is the same as the original number of variables. At this point, the total variance on all of the principal components will equal the total variance among all of the variables. In this way, all of the information contained in the original data is preserved; no information is lost: PCA is just a rotation of the data. 

The elements of an eigenvector, that is, the values within a particular row of matrix, are the weights **aij**. These values are called the **loadings**, and they describe how much each variable contributes to a particular principal component. 

	* Large loadings (+ or -) indicate that a particular variable has a strong relationship to a particular principal component. 
	* The sign of a loading indicates whether a variable and a principal component are positively or negatively correlated.


Let's go back to our **wine** example and plot the **loadings** of the PCs.

.. code:: python
	
	# Load library
	from pca import pca

	# Initialize pca with default parameters
	model = pca(normalize=True)
	
	# Fit transform and include the column labels and row labels
	results = model.fit_transform(X, col_labels=col_labels, row_labels=y)
	
	# Scatter plot with loadings
	model.biplot()


First of all, we see a nice seperation of the 3 wine classes (red, orange and gray samples). In the middle of the plot we see various arrows. Each of the arrows describes its story in the Principal Components. The angle of the arrow describes the variance of the variable that is seen in the particular PC. The length describes the strength of the loading. 

.. image:: ../figs/wine_biplot.png
   :width: 600
   :align: center


Examination of the loadings
******************************

Let's examine the **loadings** (arrows) a bit more to become even more aware what is going on in the distribution of samples given the variables. The variable **flavanoids** has a positive loading and explaines mostly the variance in the first PC1 (it is almost a horizontal line). If we would color the samples in the scatter plot based on **flavanoids** values, we expect to see a distinction between samples that are respectively left and right side of the scatter plot. 

.. code:: python
	
	# Grap the values for flavanoids
	X_feat = X[:, np.array(col_labels)=='flavanoids']

	# Color based on mean
	color_label = (X_feat>=np.mean(X_feat)).flatten()

	# Scatter based on discrete color
	model.scatter(y=color_label, title='Color on flavanoids (Gray colored samples are > mean)')

	# 3d scatter plot
	model.scatter3d(y=color_label, title='Color on flavanoids (Gray colored samples are > mean)')


.. |figA1| image:: ../figs/wine_flavanoids.png
.. |figA2| image:: ../figs/wine_flavanoids3d.png

.. table:: Color on flavanoids
   :align: center

   +----------+----------+
   | |figA1|  | |figA2|  |
   +----------+----------+

Let's take another variable for demonstration purposes. The variable **alcohol** has a strong negative loading (almost vertical), and should therefoe explains mostly the 2nd PC but the angle is not exactly vertical, thus there is also some variance seen in the 1st PC. Let's color the samples based on **alcohol**.

.. code:: python

	# Grap the values for alcohol
	X_feat = X[:, np.array(col_labels)=='alcohol']

	# Color based on mean
	color_label = (X_feat>=np.mean(X_feat)).flatten()

	# Scatter based on discrete color
	model.scatter(y=color_label, title='Color on alcohol (Gray colored samples are < mean)')

	# 3d scatter plot
	model.scatter3d(y=color_label, title='Color on alcohol (Gray colored samples are < mean)')


.. |figA8| image:: ../figs/wine_alcohol.png
.. |figA9| image:: ../figs/wine_alcohol3d.png

.. table:: Color on alcohol
   :align: center

   +----------+----------+
   | |figA8|  | |figA9|  |
   +----------+----------+





Best Performing Features
**************************

Extracting the best performing features is based on the loadings of the Principal Components, which are readily computed.
The information is stored in the object itself and we can extract it as shown underneath. 

.. code:: python

	# Print the top features.
	print(model.results['topfeat'])

	#      PC                       feature   loading  type
	#     PC1                    flavanoids  0.422934  best
	#     PC2               color_intensity -0.529996  best
	#     PC3                           ash  0.626224  best
	#     PC4                    malic_acid  0.536890  best
	#     PC5                     magnesium  0.727049  best
	#     PC6                    malic_acid -0.536814  best
	#     PC7          nonflavanoid_phenols  0.595447  best
	#     PC8                           hue -0.436624  best
	#     PC9                       proline -0.575786  best
	#     PC10  od280/od315_of_diluted_wines  0.523706  best
	#     PC9                       alcohol  0.508619  weak
	#     PC3             alcalinity_of_ash  0.612080  weak
	#     PC8                 total_phenols  0.405934  weak
	#     PC6               proanthocyanins  0.533795  weak


We see that the most of the variance for the 1st PC is derived from the variable **flavanoids**. For the 2nd component, the most variance is seen in **color_intensity**, etc.


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




References
------------

* [1] https://builtin.com/data-science/step-step-explanation-principal-component-analysis
* [2] http://strata.uga.edu/8370/lecturenotes/principalComponents.html


.. raw:: html

   <hr>
   <center>
     <script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
   </center>
   <hr>
