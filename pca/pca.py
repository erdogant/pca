"""pca is a python package that performs the principal component analysis and to make insightful plots."""

# ----------------------------------
# Name        : pca.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# ----------------------------------


# %% Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import colourmap as colourmap
import os
import wget

# %% Association learning across all variables
class pca():
    def __init__(self, n_components=0.95, n_feat=25, onehot=False, normalize=False, random_state=None):
        """Initialize pca with user-defined parameters.

        Parameters
        ----------
        onehot : [Bool] optional, (default: False)
            Boolean: Set True if X is a sparse data set such as the output of a tfidf model. Many zeros and few numbers. Note this is different then a sparse matrix. Sparse data can be in a sparse matrix.
        n_components : [0,..,1] or [1,..number of samples-1], (default: 0.95)
            Number of TOP components to be returned. Values>0 are the number of components. Values<0 are the components that covers at least the percentage of variance.
            0.95: Take the number of components that cover at least 95% of variance
            k: Take the top k components
        random_state : int optional
            Random state
        normalize : bool optional
            Normalize data, Z-score (default True)

        """
        # Store in object
        self.n_components = n_components
        self.onehot = onehot
        self.normalize = normalize
        self.random_state = random_state
        self.n_feat = n_feat

    # Make PCA fit_transform
    def fit_transform(self, X, row_labels=None, col_labels=None, verbose=3):
        """Fit PCA on data.

        Parameters
        ----------
        X : numpy array
            [NxM] array with columns as features and rows as samples.
        row_labels : [list of integers or strings] optional
            Used for colors
        col_labels : [list of string] optional
            Numpy Vector of strings: Name of the features that represent the data features and loadings

        Returns
        -------
        dict.
        loadings : pd.DataFrame
            Structured dataframe containing loadings for PCs
        X : array-like
            Reduced dimentionsality space, the Principal Components (PCs)
        explained_var : array-like
            Explained variance for each fo the PCs (same ordering as the PCs)
        model_pca : object
            Model to be used for further usage of the model.
        topn : int
            Top n components
        pcp : int
            pcp
        col_labels : array-like
            Name of the features
        y : array-like
            Determined class labels

        Examples
        --------
        >>> # Load example data
        >>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
        >>> # Fit using PCA
        >>> results = model.fit_transform(X)
        >>> # Make plots
        >>> model.scatter()
        >>> ax = model.plot()
        >>> ax = model.biplot()
        >>> 3D plots
        >>> model.scatter3d()
        >>> ax = model.biplot3d()
        >>> # Normalize out PCs
        >>> X_norm = pca.norm(X)

        """
        # Pre-processing
        if verbose>=3: print('[pca] >The PCA reduction is performed on the [%.d] columns of the input dataframe.' %(X.shape[1]))
        X, row_labels, col_labels = self._preprocessing(X, row_labels, col_labels, verbose=verbose)

        if self.n_components<1:
            pcp = self.n_components
            # Run with all components to get all PCs back. This is needed for the step after.
            model_pca, PC, loadings, percentExplVar = _explainedvar(X, n_components=None, onehot=self.onehot, random_state=self.random_state)
            # Take number of components with minimal [n_components] explained variance
            self.n_components = np.min(np.where(percentExplVar >= self.n_components)[0]) + 1
            if verbose>=3: print('[pca] >Number of components is [%d] that covers the [%.2f%%] explained variance.' %(self.n_components, pcp*100))
        else:
            model_pca, PC, loadings, percentExplVar = _explainedvar(X, n_components=self.n_components, onehot=self.onehot, random_state=self.random_state)
            pcp = percentExplVar[np.minimum(len(percentExplVar)-1, self.n_components)]

        # Combine components relations with features.
        loadings = self._postprocessing(model_pca, loadings, col_labels, self.n_components, verbose=verbose)
        # Top scoring n_components.
        topfeat = self.top_feat(loadings=loadings, n_feat=self.n_feat, verbose=verbose)
        # Store
        self.results = _store(PC, loadings, percentExplVar, model_pca, self.n_components, pcp, col_labels, row_labels, topfeat)
        # Return
        return(self.results)


    # Post processing.
    def _postprocessing(self, model_pca, loadings, col_labels, n_components, verbose=3):
        PCzip = list(zip(['PC'] * model_pca.components_.shape[0], np.arange(1,model_pca.components_.shape[0] + 1).astype(str)))
        PCnames = list(map(lambda x: ''.join(x), PCzip))
        loadings = pd.DataFrame(loadings, columns=col_labels, index=PCnames)
        # Return
        return(loadings)


    # Top scoring features
    # def top_feat(self, loadings=None, n_feat=10, verbose=3):
    #     """Compute which features explain most of the variance.
    
    #     Parameters
    #     ----------
    #     loadings : pd.DataFrame
    #         Array containing loadings.
    #     n_feat : int, default: 10
    #         Number of features that explain the space the most, dervied from the loadings.
    #     verbose : int, default: 3.
    #         Print message to screen. 
    
    #     Returns
    #     -------
    #     dataFrame containing the top ranked features with the weights.
    
    #     """
    #     if (loadings is None):
    #         try:
    #             loadings = self.results['loadings']
    #         except:
    #             raise Exception('[pca] >Error: loadings is not defined. Tip: run fit_transform() or provide the loadings yourself as input argument.') 

    #     if verbose>=3: print('[pca] >Computing best explaining features for the 1st and 2nd component.')
    #     n_feat = np.maximum(np.minimum(n_feat, loadings.shape[1]), 2)
    #     feat_weights = loadings.iloc[0:1,:].max(axis=0).sort_values(ascending=False)

    #     # Top scoring for 1st and seocnd component
    #     if loadings.shape[0]>=2:
    #         feat_weights = loadings.iloc[0:2,:].max(axis=0).sort_values(ascending=False)

    #     topfeat = feat_weights[0:n_feat]
    #     # Return
    #     return topfeat


    # Top scoring components
    def top_feat(self, loadings=None, n_feat=10, verbose=3):
        if (loadings is None):
            try:
                loadings = self.results['loadings']
            except:
                raise Exception('[pca] >Error: loadings is not defined. Tip: run fit_transform() or provide the loadings yourself as input argument.') 

        n_feat = np.maximum(np.minimum(n_feat, loadings.shape[1]), 2)
        # Top scoring for 1st component
        I1 = np.argsort(np.abs(loadings.iloc[0,:]))
        I1 = I1[::-1]
        # L1_weights = loadings.iloc[0,I1]
    
        if loadings.shape[0]>=2:
            # Top scoring for 2nd component
            I2=np.argsort(np.abs(loadings.iloc[1,:]))
            I2=I2[::-1]
            # L2_weights = loadings.iloc[0,I2]
            # Take only top loadings
            I1=I1[0:n_feat]
            I2=I2[0:n_feat]
            I = np.append(I1, I2)
        else:
            I=I1
        # Unique without sort:
        indices = np.unique(I,return_index=True)[1]
        # feat_weights = loadings.iloc[0:1,I].T
        # topfeat = feat_weights[0:n_feat]
        I = [I[index] for index in sorted(indices)]
        topfeat = loadings.iloc[0:2,I].T
        topfeat.columns = topfeat.columns.values+'_weights'
        # topfeat = topfeat.iloc[0:n_feat,:]
        return topfeat


    # Check input values
    def _preprocessing(self, X, row_labels, col_labels, verbose=3):
        if self.n_components is None:
            self.n_components = X.shape[1] - 1
            if verbose>=3: print('[pca] >n_components is set to %d' %(self.n_components))

        self.n_feat = np.min([self.n_feat, X.shape[1]])
        
        if (not self.onehot) and (not self.normalize) and (str(X.values.dtype)=='bool'):
            if verbose>=2: print('[pca] >Warning: Sparse or one-hot boolean input data is detected, it is highly recommended to set onehot=True or alternatively, normalize=True')

        # if sp.issparse(X):
            # if verbose>=1: print('[PCA] Error: A sparse matrix was passed, but dense data is required for method=barnes_hut. Use X.toarray() to convert to a dense numpy array if the array is small enough for it to fit in memory.')
        if isinstance(X, pd.DataFrame):
            if verbose>=3: print('[pca] >Processing dataframe..')
            col_labels = X.columns.values
            row_labels = X.index.values
            X = X.values
        if sp.issparse(X) and self.normalize:
            if verbose>=3: print('[pca] >Can not normalize a sparse matrix. Normalize is set to [False]')
            self.normalize=False
        if col_labels is None or len(col_labels)==0 or len(col_labels)!=X.shape[1]:
            if verbose>=3: print('[pca] >Column labels are auto-completed.')
            col_labels = np.arange(1,X.shape[1] + 1).astype(str)
        if row_labels is None or len(row_labels)!=X.shape[0]:
            row_labels=np.ones(X.shape[0])
            if verbose>=3: print('[pca] >Row labels are auto-completed.')
        if isinstance(row_labels, list):
            row_labels=np.array(row_labels)
        if isinstance(col_labels, list):
            col_labels=np.array(col_labels)
        if (sp.issparse(X) is False) and (self.n_components > X.shape[1]):
            raise Exception('[pca] >Number of components can not be more then number of features.')
    
        # normalize data
        if self.normalize:
            if verbose>=3: print('[pca] >Normalizing input data per feature (zero mean)..')
            # Plot the data distribution
            # fig,(ax1,ax2)=plt.subplots(1,2, figsize=(15,5))
            # ax1.hist(X.ravel().astype(float), bins=50)
            # ax1.set_ylabel('frequency')
            # ax1.set_xlabel('Values')
            # ax1.set_title('RAW')
            # ax1.grid(True)

            X = preprocessing.scale(X, with_mean=True, with_std=True, axis=0)

            # Plot the data distribution
            # ax2.hist(X.ravel().astype(float), bins=50)
            # ax2.set_ylabel('frequency')
            # ax2.set_xlabel('Values')
            # ax2.set_title('Zero-mean with unit variance normalized')
            # ax2.grid(True)

        return(X, row_labels, col_labels)


    # Figure pre processing
    def _fig_preprocessing(self, y, n_feat):
        if hasattr(self, 'PC'): raise Exception('[pca] >Error: Principal components are not derived yet. Tip: run fit_transform() first.')
        if self.results['PC'].shape[1]<1: raise Exception('[pca] >Requires at least 1 PC to make plot.')


        if (n_feat is not None):
            topfeat = self.top_feat(n_feat=n_feat)
            n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[1]), 2)
        else:
            topfeat = self.results['topfeat']
            n_feat = self.n_feat

        if (y is not None):
            if len(y)!=self.results['PC'].shape[0]: raise Exception('[pca] >Error: Input variable [y] should have some length as the number input samples: [%d].' %(self.results['PC'].shape[0]))
            y = y.astype(str)
        else:
            y = self.results['PC'].index.values.astype(str)

        if len(self.results['explained_var'])<=1:
            raise Exception('[pca] >Error: No PCs are found with explained variance..')
                                                                                                                
        return y, topfeat, n_feat


    # Scatter plot
    def scatter3d(self, y=None, figsize=(10,8)):
        if self.results['PC'].shape[1]>=3:
            fig, ax = self.scatter(y=y, d3=True, figsize=figsize)
        else:
            print('[pca] >Error: There are not enough PCs to make a 3d-plot.')
            fig, ax = None, None
        return fig, ax


    # Scatter plot
    def scatter(self, y=None, d3=False, figsize=(10,8)):
        [fig,ax]=plt.subplots(figsize=figsize, edgecolor='k')

        if y is None:
            y, _, _ = self._fig_preprocessing(y, None)

        # Get coordinates
        xs = self.results['PC'].iloc[:,0].values
        if self.results['PC'].shape[1]>1:
            ys = self.results['PC'].iloc[:,1].values
        else:
            ys = np.zeros(len(xs))
        if d3:
            zs = self.results['PC'].iloc[:,2].values
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # Make scatter plot
        uiy = np.unique(y)
        getcolors = np.array(colourmap.generate(len(uiy)))
        for i,yk in enumerate(uiy):
            Iloc = (yk==y)
            if d3:
                ax.scatter(xs[Iloc],ys[Iloc],zs[Iloc],color=getcolors[i,:], s=25)
            else:
                ax.scatter(xs[Iloc], ys[Iloc], color=getcolors[i,:], s=25)
                ax.annotate(yk, (np.mean(xs[Iloc]), np.mean(ys[Iloc])))
    
        # Set y
        ax.set_xlabel('PC1 ('+ str(self.results['model'].explained_variance_ratio_[0] * 100)[0:4] + '% expl.var)')
        ax.set_ylabel('PC2 ('+ str(self.results['model'].explained_variance_ratio_[1] * 100)[0:4] + '% expl.var)')
        if d3: ax.set_zlabel('PC3 ('+ str(self.results['model'].explained_variance_ratio_[2] * 100)[0:4] + '% expl.var)')
        ax.set_title(str(self.n_components)+' Principal Components explain [' + str(self.results['pcp']*100)[0:5] + '%] of the variance')
        ax.grid(True)

        return fig, ax

    # biplot
    def biplot(self, y=None, n_feat=None, figsize=(10,8)):
        """Create the Biplot based on model.

        Parameters
        ----------
        figsize : (float, float), optional, default: None
            (width, height) in inches. If not provided, defaults to rcParams["figure.figsize"] = (10,8)

        Returns
        -------
        tuple containing (fig, ax)

        """
        # Pre-processing
        y, topfeat, n_feat = self._fig_preprocessing(y, n_feat)
        # Figure
        fig, ax  = self.scatter(y=y, figsize=figsize)

        # Gather loadings from the top features from topfeat
        xvector = self.results['loadings'][topfeat.index.values].iloc[0,:]
        yvector = self.results['loadings'][topfeat.index.values].iloc[1,:]
        # Use the PCs only for scaling purposes
        xs = self.results['PC'].iloc[:,0].values
        ys = self.results['PC'].iloc[:,1].values
        # Boundaries figures
        maxR = np.max(xs)*0.8
        maxL = np.min(xs)*0.8
        maxT = np.max(ys)*0.8
        maxB = np.min(ys)*0.8

        np.where(np.logical_and(np.sign(xvector)>0, (np.sign(yvector)>0)))

        # Plot and scale values for arrows and text
        scalex = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[0,:].max() - self.results['loadings'][topfeat.index.values].iloc[0,:].min())
        scaley = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[1,:].max() - self.results['loadings'][topfeat.index.values].iloc[1,:].min())
        # Plot the arrows
        for i in range(0, n_feat):
            # arrows project features (ie columns from csv) as vectors onto PC axes
            newx = xvector[i] * scalex
            newy = yvector[i] * scaley
            # figscaling = np.abs([np.abs(xs).max() / newx, np.abs(ys).max() / newy])
            # figscaling = figscaling.max()
            # newx = newx * figscaling * 0.1
            # newy = newy * figscaling * 0.1
            newx = newx * 500
            newy = newy * 500

            # Max boundary right x-axis
            if np.sign(newx)>0:
                newx = np.minimum(newx, maxR)
            # Max boundary left x-axis
            if np.sign(newx)<0:
                newx = np.maximum(newx, maxL)
            # Max boundary Top
            if np.sign(newy)>0:
                newy = np.minimum(newy, maxT)
            # Max boundary Bottom
            if np.sign(newy)<0:
                newy = np.maximum(newy, maxB)
            
            ax.arrow(0, 0, newx, newy, color='r', width=0.005, head_width=0.05, alpha=0.6)
            ax.text(newx * 1.25, newy * 1.25, xvector.index.values[i], color='red', ha='center', va='center')
    
        plt.show()
        return(fig, ax)


    # biplot3d
    def biplot3d(self, y=None, n_feat=None, figsize=(10,8)):
        """Make biplot in 3d.
    
        Parameters
        ----------
        model : dict
            model created by the fit() function.
        figsize : (float, float), optional, default: None
            (width, height) in inches. If not provided, defaults to rcParams["figure.figsize"] = (10,8)
    
        Returns
        -------
        tuple containing (fig, ax)
    
        """

        if self.results['PC'].shape[1]<3:
            print('[pca] >Requires 3 PCs to make 3d plot. Try to use biplot() instead.')
            return None, None

        # Pre-processing
        y, topfeat, n_feat = self._fig_preprocessing(y, n_feat)
        # Figure
        fig, ax = self.scatter(y=y, d3=True, figsize=figsize)

        # Gather top N loadings
        xvector = self.results['loadings'][topfeat.index.values].iloc[0,:]
        yvector = self.results['loadings'][topfeat.index.values].iloc[1,:]
        zvector = self.results['loadings'][topfeat.index.values].iloc[2,:]
        xs = self.results['PC'].iloc[:,0].values
        ys = self.results['PC'].iloc[:,1].values
        zs = self.results['PC'].iloc[:,2].values

        # Plot and scale values for arrows and text
        scalex = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[0,:].max() - self.results['loadings'][topfeat.index.values].iloc[0,:].min())
        scaley = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[1,:].max() - self.results['loadings'][topfeat.index.values].iloc[1,:].min())
        scalez = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[2,:].max() - self.results['loadings'][topfeat.index.values].iloc[2,:].min())

        # Plot the arrows
        for i in range(0, n_feat):
            # arrows project features (ie columns from csv) as vectors onto PC axes
            newx=xvector[i] * scalex
            newy=yvector[i] * scaley
            newz=zvector[i] * scalez
            figscaling = np.abs([np.abs(xs).max() / newx, np.abs(ys).max() / newy])
            figscaling = figscaling.min()
            newx=newx * figscaling * 0.5
            newy=newy * figscaling * 0.5
            newz=newz * figscaling * 0.5
            # ax.arrow(0, 0, newx, newy, color='r', width=0.005, head_width=0.05, alpha=0.6)
            ax.text(newx, newy, newz, xvector.index.values[i], color='black', ha='center', va='center')
            # a = Arrow3D([0, newx], [0, newy], [0, newz], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

        plt.show()
        return(fig, ax)


    # Show explained variance plot
    def plot(self, n_components=None, figsize=(10,8), xsteps=None):
        """Make plot.

        Parameters
        ----------
        model : dict
            model created by the fit() function.
        figsize : (float, float), optional, default: None
            (width, height) in inches. If not provided, defaults to rcParams["figure.figsize"] = (10,8)

        Returns
        -------
        tuple containing (fig, ax)
    
        """
        if n_components is not None:
            explvarCum = self.results['explained_var'][0:n_components]
            explvar = self.results['model'].explained_variance_ratio_[0:n_components]
        else:
            explvarCum = self.results['explained_var']
            explvar = self.results['model'].explained_variance_ratio_
        xtick_idx = np.arange(1,len(explvar) + 1)

        # Make figure
        fig,ax = plt.subplots(figsize=figsize, edgecolor='k')
        plt.plot(xtick_idx, explvarCum, 'o-', color='k', linewidth=1, label='Cumulative explained variance')

        # Set xticks if less then 100 datapoints
        if len(explvar)<100:
            ax.set_xticks(xtick_idx)
            xticklabel=xtick_idx.astype(str)
            if xsteps is not None:
                xticklabel[np.arange(1,len(xticklabel),xsteps)]=''
            ax.set_xticklabels(xticklabel, rotation=90, ha='left', va='top')

        plt.ylabel('Percentage explained variance')
        plt.xlabel('Principle Component')
        plt.ylim([0, 1.05])
        plt.xlim([0, len(explvar) + 1])
        titletxt = 'Cumulative explained variance\n ' + str(self.n_components) + ' Principal Components explain [' + str(self.results['pcp']*100)[0:5] + '%] of the variance.'
        plt.title(titletxt)
        plt.grid(True)

        # Plot vertical line To stress the cut-off point
        ax.axvline(self.n_components, linewidth=0.8, color='r')
        ax.axhline(y=self.results['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
        if len(xtick_idx)<100:
            plt.bar(xtick_idx, explvar, color='#3182bd', alpha=0.8, label='Explained variance')
        plt.show()
        plt.draw()
        return(fig, ax)


    # Top scoring components
    def norm(self, X, n_components=None, pcexclude=[1]):
        """Normalize out PCs.
    
        Normalize your data using the principal components.
        As an example, suppose there is (technical) variation in the fist
        component and you want that out. This function transforms the data using
        the components that you want, e.g., starting from the 2nd pc, up to the
        pc that contains at least 95% of the explained variance
    
    
        Parameters
        ----------
        X : numpy array
            Data set.
        n_components : float [0..1], optional
            Number of PCs to keep based on the explained variance. The default is 1 (keeping all)
        pcexclude : list of int, optional
            The PCs to exclude. The default is [1].
    
        Returns
        -------
        Normalized numpy array.
    
        """
        if n_components is None:
            self.n_components = X.shape[1]
        else:
            self.n_components = n_components

        if not isinstance(pcexclude,list): pcexclude=[pcexclude]

        # Fit using PCA
        _ = self.fit_transform(X)
        coeff = self.results['loadings'].values
        score = self.results['PC']
        # Compute explained percentage of variance
        q = self.results['explained_var']
        ndims = np.where(q<=self.n_components)[0]
        ndims = (np.setdiff1d(ndims + 1,pcexclude)) - 1
        # Transform data
        out = np.repeat(np.mean(X.values, axis=1).reshape(-1,1), X.shape[1], axis=1) + np.dot(score.values[:,ndims], coeff[:,ndims].T)
        # Return
        return(out)

    # Import example
    def import_example(self, data='titanic', verbose=3):
        """Import example dataset from github source.
    
        Parameters
        ----------
        data : str, optional
            Name of the dataset 'sprinkler' or 'titanic' or 'student'.
        verbose : int, optional
            Print message to screen. The default is 3.
    
        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.
    
        """
        return import_example(data=data, verbose=verbose)

# %% Explained variance
def _explainedvar(X, n_components=None, onehot=False, random_state=None, n_jobs=-1, verbose=3):
    # Create the model
    if sp.issparse(X):
        if verbose>=3: print('[pca] >Fiting using Truncated SVD..')
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif onehot:
        if verbose>=3: print('[pca] >Fitting using Sparse PCA..')
        model = SparsePCA(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
    else:
        if verbose>=3: print('[pca] >Fitting using PCA..')
        model = PCA(n_components=n_components, random_state=random_state)

    # Fit model
    model.fit(X)
    # Do the reduction
    if verbose>=3: print('[pca] >Computing loadings and PCs..')
    loadings = model.components_ # Ook wel de coeeficienten genoemd: coefs!
    PC = model.transform(X)
    if not onehot:
        # Compute explained variance, top 95% variance
        if verbose>=3: print('[pca] >Computing explained variance..')
        percentExplVar = model.explained_variance_ratio_.cumsum()
    else:
        percentExplVar = None
    # Return
    return(model, PC, loadings, percentExplVar)


# %% Store results
def _store(PC, loadings, percentExplVar, model_pca, n_components, pcp, col_labels, row_labels, topfeat):
    out = {}
    out['loadings'] = loadings
    out['PC'] = pd.DataFrame(data=PC[:,0:n_components], index=row_labels, columns=loadings.index.values[0:n_components])
    out['explained_var'] = percentExplVar
    out['model'] = model_pca
    out['pcp'] = pcp
    out['topfeat'] = topfeat
    return out


# %% Import example dataset from github.
def import_example(data='titanic', verbose=3):
    """Import example dataset from github source.

    Parameters
    ----------
    data : str, optional
        Name of the dataset 'sprinkler' or 'titanic' or 'student'.
    verbose : int, optional
        Print message to screen. The default is 3.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if data=='sprinkler':
        url='https://erdogant.github.io/datasets/sprinkler.zip'
    elif data=='titanic':
        url='https://erdogant.github.io/datasets/titanic_train.zip'
    elif data=='student':
        url='https://erdogant.github.io/datasets/student_train.zip'

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.mkdir(curpath)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[pca] >Downloading example dataset from github source..')
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[pca] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA)
    # Return
    return df
