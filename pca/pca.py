"""pca: A Python Package for Principal Component Analysis."""

# %% Libraries
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import scatterd as scatterd
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD  # MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text
import statsmodels.stats.multitest as multitest


# %% Association learning across all variables
class pca:
    """pca module.

    Parameters
    ----------
    n_components : [0..1] or [1..number of samples-1], (default: 0.95)
        Number of PCs to be returned. When n_components is set >0, the specified number of PCs is returned.
        When n_components is set between [0..1], the number of PCs is returned that covers at least this percentage of variance.
        n_components=None : Return all PCs
        n_components=0.95 : Return the number of PCs that cover at least 95% of variance.
        n_components=3    : Return the top 3 PCs.
    n_feat : int, default: 10
        Number of features that explain the space the most, dervied from the loadings. This parameter is used for vizualization purposes only.
    method : 'pca' (default)
        'pca' : Principal Component Analysis.
        'sparse_pca' : Sparse Principal Components Analysis.
        'trunc_svd' : truncated SVD (aka LSA).
    alpha : float, default: 0.05
        Alpha to set the threshold to determine the outliers based on on the Hoteling T2 test.
    multipletests : str, default: 'fdr_bh'
        Multiple testing method to correct for the Hoteling T2 test:
            * None : No multiple testing
            * 'bonferroni' : one-step correction
            * 'sidak' : one-step correction
            * 'holm-sidak' : step down method using Sidak adjustments
            * 'holm' : step-down method using Bonferroni adjustments
            * 'simes-hochberg' : step-up method  (independent)
            * 'hommel' : closed method based on Simes tests (non-negative)
            * 'fdr_bh' : Benjamini/Hochberg  (non-negative)
            * 'fdr_by' : Benjamini/Yekutieli (negative)
            * 'fdr_tsbh' : two stage fdr correction (non-negative)
            * 'fdr_tsbky' : two stage fdr correction (non-negative
    n_std : int, default: 3
        Number of standard deviations to determine the outliers using SPE/DmodX method.
    onehot : [Bool] optional, (default: False)
        Boolean: Set True if X is a sparse data set such as the output of a tfidf model. Many zeros and few numbers.
        Note this is different then a sparse matrix. In case of a sparse matrix, use method='trunc_svd'.
    normalize : bool (default : False)
        Normalize data, Z-score
    detect_outliers : list (default : ['ht2','spe'])
        None: Do not compute outliers.
        'ht2': compute outliers based on Hotelling T2.
        'spe': compute outliers basedon SPE/DmodX method.
    random_state : int optional
        Random state
    Verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    References
    ----------
    * Blog: https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559
    * Github: https://github.com/erdogant/pca
    * Documentation: https://erdogant.github.io/pca/

    """

    def __init__(self, n_components=0.95, n_feat=25, method='pca', alpha=0.05, multipletests='fdr_bh', n_std=3, onehot=False, normalize=False, detect_outliers=['ht2', 'spe'], random_state=None, verbose=3):
        """Initialize pca with user-defined parameters."""
        if isinstance(detect_outliers, str): detect_outliers = [detect_outliers]
        if onehot:
            if verbose>=3: print('[pca] >Method is set to: [sparse_pca] because onehot=True.')
            method = 'sparse_pca'

        # Store in object
        self.n_components = n_components
        self.method = method.lower()
        self.onehot = onehot
        self.normalize = normalize
        self.random_state = random_state
        self.n_feat = n_feat
        self.alpha = alpha
        self.multipletests = multipletests
        self.n_std = n_std
        self.detect_outliers = detect_outliers
        self.verbose = verbose

    # Make PCA fit_transform
    def transform(self, X, row_labels=None, col_labels=None, update_outlier_params=True, verbose=None):
        """Transform new input data with fitted model.

        Parameters
        ----------
        X : array-like : Can be of type Numpy or DataFrame
            [NxM] array with columns as features and rows as samples.
        update_outlier_params : bool (default: True)
            True : Update the parameters for outlier detection so that the model learns from the new unseen input. This will cause that some initial outliers may not be an outlier anymore after a certain point.
            False: Do not update outlier parameters and outliers that were initially detected, will always stay an outlier.
        row_labels : [list of integers or strings] optional
            Used for colors.
        col_labels : [list of string] optional
            Numpy or list of strings: Name of the features that represent the data features and loadings. This should match the number of columns in the data. Use this option when using a numpy-array. For a pandas-dataframe, the column names are used but are overruled when using this parameter.
        Verbose : int (default : 3)
            Set verbose during initialization.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_iris
        >>> import pandas as pd
        >>> from pca import pca
        >>>
        >>> # Initialize
        >>> model = pca(n_components=2, normalize=True)
        >>> # Dataset
        >>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
        >>>
        >>> # Gather some random samples across the classes.
        >>> idx=[0,1,2,3,4,50,51,52,53,54,55,100,101,102,103,104,105]
        >>> X_unseen = X.iloc[idx, :]
        >>>
        >>> # Label the unseen samples differently.
        >>> X.index.values[idx]=3
        >>>
        >>> # Fit transform
        >>> model.fit_transform(X)
        >>>
        >>> # Transform the "unseen" data with the fitted model. Note that these datapoints are not really unseen as they are readily fitted above.
        >>> # But for the sake of example, you can see that these samples will be transformed exactly on top of the orignial ones.
        >>> PCnew = model.transform(X_unseen)
        >>>
        >>> # Plot PC space
        >>> model.scatter()
        >>> # Plot the new "unseen" samples on top of the existing space
        >>> plt.scatter(PCnew.iloc[:, 0], PCnew.iloc[:, 1], marker='x')

        Returns
        -------
        pca transformed data.

        """
        if verbose is None: verbose = self.verbose
        # Check type to make sure we can perform matrix operations
        if isinstance(X, list):
            X = np.array(X)
        if row_labels is None:
            row_labels = np.repeat('mapped', X.shape[0])
        # Pre-processing using scaler.
        X_scaled, row_labels, _, _ = self._preprocessing(X, row_labels, col_labels, scaler=self.results['scaler'], verbose=verbose)
        # Transform the data using fitted model.
        PCs = self.results['model'].transform(X_scaled)
        # Store in dataframe
        columns = ['PC{}'.format(i + 1) for i in np.arange(0, PCs.shape[1])]
        PCs = pd.DataFrame(data=PCs, index=row_labels, columns=columns)

        # Add mapped PCs to dataframe
        if self.detect_outliers is not None:
            # By setting the outliers params to None, it will update the parameters on the new input data.
            if update_outlier_params:
                self.results['outliers_params']['paramT2']=None
                self.results['outliers_params']['paramSPE']=None
            PCtot = pd.concat([self.results['PC'], PCs], axis=0)
            # Detection of outliers
            self.results['outliers'], _ = self.compute_outliers(PCtot, verbose=verbose)
            # Store
            self.results['PC'] = PCtot

        # Return
        return PCs

    # Make PCA fit_transform
    def fit_transform(self, X, row_labels=None, col_labels=None, verbose=None):
        """Fit PCA on data.

        Parameters
        ----------
        X : array-like : Can be of type Numpy or DataFrame
            [NxM] array with columns as features and rows as samples.
        row_labels : [list of integers or strings] optional
            Used for colors.
        col_labels : [list of string] optional
            Numpy or list of strings: Name of the features that represent the data features and loadings. This should match the number of columns in the data. Use this option when using a numpy-array. For a pandas-dataframe, the column names are used but are overruled when using this parameter.
        Verbose : int (default : 3)
            Set verbose during initialization.

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
        >>> from pca import pca
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
        >>>
        >>> Initialize
        >>> model = pca(n_components=3)
        >>> # Fit using PCA
        >>> results = model.fit_transform(X)
        >>>
        >>> # Make plots
        >>> fig, ax = model.scatter()
        >>> fig, ax = model.plot()
        >>> fig, ax = model.biplot()
        >>> fig, ax = model.biplot(SPE=True, hotellingt2=True)
        >>>
        >>> 3D plots
        >>> fig, ax = model.scatter3d()
        >>> fig, ax = model.biplot3d()
        >>> fig, ax = model.biplot3d(SPE=True, hotellingt2=True)
        >>>
        >>> # Normalize out PCs
        >>> X_norm = model.norm(X)

        """
        if verbose is None: verbose = self.verbose
        percentExplVar=None
        # Check type to make sure we can perform matrix operations
        if sp.issparse(X):
            if self.verbose>=3: print('[pca] >Input data is a sparse matrix. Method is set to: [trunc_svd].')
            self.method = 'trunc_svd'
        if isinstance(X, list):
            X = np.array(X)

        # Clean readily fitted models to ensure correct results.
        self._clean(verbose=verbose)
        # Pre-processing
        X, row_labels, col_labels, scaler = self._preprocessing(X, row_labels, col_labels, verbose=verbose)

        if self.n_components<1:
            if verbose>=3: print('[pca] >The PCA reduction is performed to capture [%.1f%%] explained variance using the [%.d] columns of the input data.' %(self.n_components * 100, X.shape[1]))
            pcp = self.n_components
            # Run with all components to get all PCs back. This is needed for the step after.
            _, _, _, percentExplVar = _explainedvar(X, method=self.method, n_components=None, onehot=self.onehot, random_state=self.random_state, verbose=verbose)
            # Take number of components with minimal [n_components] explained variance
            if percentExplVar is None:
                self.n_components = X.shape[1] - 1
                if verbose>=3: print('[pca] >n_components is set to %d' %(self.n_components))
            else:
                self.n_components = np.min(np.where(percentExplVar >= self.n_components)[0]) + 1
                if verbose>=3: print('[pca] >Number of components is [%d] that covers the [%.2f%%] explained variance.' %(self.n_components, pcp * 100))

        if verbose>=3: print('[pca] >The PCA reduction is performed on the [%.d] columns of the input dataframe.' %(X.shape[1]))
        model_pca, PC, loadings, percentExplVar = _explainedvar(X, method=self.method, n_components=self.n_components, onehot=self.onehot, random_state=self.random_state, percentExplVar=percentExplVar, verbose=verbose)
        pcp = None if percentExplVar is None else percentExplVar[np.minimum(len(percentExplVar) - 1, self.n_components)]

        # Combine components relations with features
        loadings = self._postprocessing(model_pca, loadings, col_labels, self.n_components, verbose=verbose)
        # Top scoring n_components
        topfeat = self.compute_topfeat(loadings=loadings, verbose=verbose)
        # Detection of outliers
        outliers, outliers_params = self.compute_outliers(PC, verbose=verbose)
        # Store
        self.results = _store(PC, loadings, percentExplVar, model_pca, self.n_components, pcp, col_labels, row_labels, topfeat, outliers, scaler, outliers_params)
        # Return
        return self.results

    def _clean(self, verbose=3):
        # Clean readily fitted models to ensure correct results.
        if hasattr(self, 'results'):
            if verbose>=3: print('[pca] >Cleaning previous fitted model results..')
            if hasattr(self, 'results'): del self.results

    # Outlier detection
    def compute_outliers(self, PC, n_std=3, verbose=3):
        """Compute outliers.

        Parameters
        ----------
        PC : Array-like
            Principal Components.
        n_std : int, (default: 3)
            Standard deviation. The default is 3.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        outliers : numpy array
            Array containing outliers.
        outliers_params: dict, (default: None)
            Contains parameters for hotellingsT2() and spe_dmodx(), reusable in the future.
        """
        # Convert to numpy array if required
        if isinstance(PC, pd.DataFrame): PC = np.array(PC)
        # Initialize
        outliersHT2, outliersELIPS = pd.DataFrame(), pd.DataFrame()
        if hasattr(self, 'results'):
            paramT2 = self.results['outliers_params'].get('paramT2', None)
            paramSPE = self.results['outliers_params'].get('paramSPE', None)
        else:
            paramT2, paramSPE = None, None

        if np.any(np.isin(self.detect_outliers, 'ht2')):
            # Detection of outliers using hotelling T2 test.
            if (paramT2 is not None) and (verbose>=3): print('[pca] >compute hotellingsT2 with precomputed parameter.')
            outliersHT2, _, paramT2 = hotellingsT2(PC, alpha=self.alpha, df=1, n_components=self.n_components, multipletests=self.multipletests, param=paramT2, verbose=verbose)
        if np.any(np.isin(self.detect_outliers, 'spe')):
            # Detection of outliers using elipse method.
            if (paramSPE is not None) and (verbose>=3): print('[pca] >compute SPE with precomputed parameter.')
            outliersELIPS, _, paramSPE = spe_dmodx(PC, n_std=self.n_std, param=paramSPE, verbose=verbose)
        # Combine
        outliers = pd.concat([outliersHT2, outliersELIPS], axis=1)
        outliers_params = {'paramT2': paramT2, 'paramSPE': paramSPE}
        return outliers, outliers_params

    # Post processing.
    def _postprocessing(self, model_pca, loadings, col_labels, n_components, verbose=3):
        PCzip = list(zip(['PC'] * model_pca.components_.shape[0], np.arange(1, model_pca.components_.shape[0] + 1).astype(str)))
        PCnames = list(map(lambda x: ''.join(x), PCzip))
        loadings = pd.DataFrame(loadings, columns=col_labels, index=PCnames)
        # Return
        return loadings

    # Top scoring components
    def compute_topfeat(self, loadings=None, verbose=3):
        """Compute the top-scoring features.

        Description
        -----------
        Per Principal Component, the feature with absolute maximum loading is stored.
        This can result into the detection of PCs that contain the same features. The feature that were never detected are stored as "weak".

        Parameters
        ----------
        loadings : array-like
            The array containing the loading information of the Principal Components.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        topfeat : pd.DataFrame
            Best performing features per PC.

        """
        if (loadings is None):
            try:
                # Get feature names
                initial_feature_names = self.results['loadings'].columns.values
                loadings = self.results['loadings'].values.copy()
            except:
                raise Exception('[pca] >Error: loadings is not defined. Tip: run model.fit_transform() or provide the loadings yourself as input argument.')

        if isinstance(loadings, pd.DataFrame):
            initial_feature_names = loadings.columns.values
            loadings = loadings.values

        # number of components
        n_pcs = loadings.shape[0]
        # get the index of the most important feature on EACH component
        idx = [np.abs(loadings[i]).argmax() for i in range(n_pcs)]
        # The the loadings
        loading_best = loadings[np.arange(0, n_pcs), idx]
        # get the names
        most_important_names = [initial_feature_names[idx[i]] for i in range(len(idx))]
        # Make dict with most important features
        dic = {'PC{}'.format(i + 1): most_important_names[i] for i in range(len(most_important_names))}
        # Collect the features that were never discovered. The weak features.
        idxcol = np.setdiff1d(range(loadings.shape[1]), idx)
        # get the names
        least_important_names = [initial_feature_names[idxcol[i]] for i in range(len(idxcol))]
        # Find the strongest loading across the PCs for the least important ones
        idxrow = [np.abs(loadings[:, i]).argmax() for i in idxcol]
        loading_weak = loadings[idxrow, idxcol]
        # Make dict with most important features
        # dic_weak = {'weak'.format(i+1): least_important_names[i] for i in range(len(least_important_names))}
        PC_weak = ['PC{}'.format(i + 1) for i in idxrow]

        # build the dataframe
        topfeat = pd.DataFrame(list(dic.items()), columns=['PC', 'feature'])
        topfeat['loading'] = loading_best
        topfeat['type'] = 'best'
        # Weak features
        weakfeat = pd.DataFrame({'PC': PC_weak, 'feature': least_important_names, 'loading': loading_weak, 'type': 'weak'})

        # Combine features
        df = pd.concat([topfeat, weakfeat])
        df.reset_index(drop=True, inplace=True)
        # Return
        return df

    # Check input values
    def _preprocessing(self, X, row_labels, col_labels, scaler=None, verbose=3):
        if self.n_components is None:
            self.n_components = X.shape[1] - 1
            if verbose>=3: print('[pca] >n_components is set to %d' %(self.n_components))

        self.n_feat = np.min([self.n_feat, X.shape[1]])

        if (not self.onehot) and (not self.normalize) and isinstance(X, pd.DataFrame) and (str(X.values.dtype)=='bool'):
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
            col_labels = np.arange(1, X.shape[1] + 1).astype(str)
        if row_labels is None or len(row_labels)!=X.shape[0]:
            row_labels=np.ones(X.shape[0])
            if verbose>=3: print('[pca] >Row labels are auto-completed.')
        if isinstance(row_labels, list):
            row_labels=np.array(row_labels)
        if isinstance(col_labels, list):
            col_labels=np.array(col_labels)
        if (sp.issparse(X) is False) and (self.n_components > X.shape[1]):
            # raise Exception('[pca] >Number of components can not be more then number of features.')
            if verbose>=2: print('[pca] >Warning: >Number of components can not be more then number of features. n_components is set to %d' %(X.shape[1] - 1))
            self.n_components = X.shape[1] - 1

        # normalize data
        if self.normalize:
            if verbose>=3: print('[pca] >Normalizing input data per feature (zero mean and unit variance)..')
            # Plot the data distribution
            # fig,(ax1,ax2)=plt.subplots(1,2, figsize=(15,5))
            # ax1.hist(X.ravel().astype(float), bins=50)
            # ax1.set_ylabel('frequency')
            # ax1.set_xlabel('Values')
            # ax1.set_title('RAW')
            # ax1.grid(True)

            # X = preprocessing.scale(X, with_mean=True, with_std=True, axis=0)

            # IF the scaler is not yet fitted, make scaler object.
            if scaler is None:
                scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
            X = scaler.transform(X)

            # Plot the data distribution
            # ax2.hist(X.ravel().astype(float), bins=50)
            # ax2.set_ylabel('frequency')
            # ax2.set_xlabel('Values')
            # ax2.set_title('Zero-mean with unit variance normalized')
            # ax2.grid(True)

        return (X, row_labels, col_labels, scaler)

    # Figure pre processing
    def _fig_preprocessing(self, y, n_feat, d3):
        if hasattr(self, 'PC'): raise Exception('[pca] >Error: Principal components are not derived yet. Tip: run fit_transform() first.')
        if self.results['PC'].shape[1]<1: raise Exception('[pca] >Requires at least 1 PC to make plot.')

        if (n_feat is not None):
            topfeat = self.compute_topfeat()
            # n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[0]), 2)
        else:
            topfeat = self.results['topfeat']
            n_feat = self.n_feat

        if d3:
            n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[1]), 3)
        else:
            n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[1]), 2)

        if (y is not None):
            if len(y)!=self.results['PC'].shape[0]: raise Exception('[pca] >Error: Input variable [y] should have some length as the number input samples: [%d].' %(self.results['PC'].shape[0]))
            y = y.astype(str)
        else:
            y = self.results['PC'].index.values.astype(str)

        if self.method=='sparse_pca':
            print('[pca] >sparse pca does not supported variance ratio and therefore, biplots will not be supported. <return>')
            self.results['explained_var'] = [None, None]
            self.results['model'].explained_variance_ratio_ = [0, 0]
            self.results['pcp'] = 0

        if (self.results['explained_var'] is None) or len(self.results['explained_var'])<=1:
            raise Exception('[pca] >Error: No PCs are found with explained variance.')

        return y, topfeat, n_feat

    # Scatter plot
    def scatter3d(self,
                  y=None,
                  c=None,
                  s=50,
                  marker='.',
                  jitter=None,
                  label=True,
                  PC=[0, 1, 2],
                  SPE=False,
                  hotellingt2=False,
                  alpha_transparency=1,
                  gradient=None,
                  fontdict={'weight': 'normal', 'size': 12, 'ha': 'center', 'va': 'center', 'c': 'black'},
                  cmap='Set1',
                  title=None,
                  legend=True,
                  figsize=(15, 10),
                  visible=True,
                  fig=None,
                  ax=None,
                  verbose=None):
        """Scatter 3d plot.

        Parameters
        ----------
        y : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            Color of samples in RGB colors.
            [0,0,0]: If a single color is given, all samples get that color.
        s: Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: '.').
            Marker for the samples.
            '.' : All data points get this marker
            ['.', '*', 's', ..]: Specify per sample the marker type.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        label : Bool, default: True
            True Show the labels.
            False: Do not show the labels
            None: Ignore all labels (this will significanly speed up the scatterplot)
        PC : list, default : [0, 1, 2]
            Plot the first three Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
        hotellingt2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
        alpha_transparency: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex (ending) color for the gradient of the scatterplot colors.
            '#FFFFFF'
        fontdict : dict.
            dictionary containing properties for the arrow font-text
            {'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': 'black'}
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : Bool, default: True
            Show the legend based on the unique y-labels.
        figsize : (int, int), optional, default: (15, 10)
            (width, height) in inches.
        visible : Bool, default: True
            Visible status of the Figure. When False, figure is created on the background.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        tuple containing (fig, ax)

        """
        if verbose is None: verbose = self.verbose
        if self.results['PC'].shape[1]>=3:
            fig, ax = self.scatter(y=y,
                                   c=c,
                                   s=s,
                                   marker=marker,
                                   jitter=jitter,
                                   d3=True,
                                   label=label,
                                   PC=PC, SPE=SPE,
                                   hotellingt2=hotellingt2,
                                   alpha_transparency=alpha_transparency,
                                   gradient=gradient,
                                   fontdict=fontdict,
                                   cmap=cmap,
                                   title=title,
                                   legend=legend,
                                   figsize=figsize,
                                   visible=visible,
                                   fig=fig,
                                   ax=ax,
                                   verbose=verbose)
        else:
            print('[pca] >Error: There are not enough PCs to make a 3d-plot.')
            fig, ax = None, None
        return fig, ax

    # Scatter plot
    def scatter(self,
                y=None,
                c=None,
                s=50,
                marker='.',
                jitter=None,
                d3=False,
                label=True,
                PC=[0, 1],
                SPE=False,
                hotellingt2=False,
                alpha_transparency=1,
                gradient=None,
                fontdict={'weight': 'normal', 'size': 12, 'ha': 'center', 'va': 'center', 'c': 'black'},
                cmap='Set1',
                title=None,
                legend=True,
                figsize=(20, 15),
                visible=True,
                fig=None,
                ax=None,
                verbose=3):
        """Scatter 2d plot.

        Parameters
        ----------
        y : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            Color of samples in RGB colors.
            [0,0,0]: If a single color is given, all samples get that color.
        s: Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: '.').
            Marker for the samples.
            '.' : All data points get this marker
            ['.', '*', 's', ..]: Specify per sample the marker type.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        d3 : Bool, default: False
            3d plot is created when True.
        label : Bool, default: True
            True Show the labels.
            False: Do not show the labels
            None: Ignore all labels (this will significanly speed up the scatterplot)
        PC : list, default : [0, 1]
            Plot the first two Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
        hotellingt2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
        alpha_transparency: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex color to make a lineair gradient for the scatterplot.
            '#FFFFFF'
        fontdict : dict.
            dictionary containing properties for the arrow font-text
            {'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': 'black'}
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : Bool, default: True
            Show the legend based on the unique y-labels.
        figsize : (int, int), optional, default: (15, 10)
            (width, height) in inches.
        visible : Bool, default: True
            Visible status of the Figure. When False, figure is created on the background.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        tuple containing (fig, ax)

        """
        if verbose is None: verbose = self.verbose
        if not hasattr(self, 'results'):
            if verbose>=2: print('[pca]> No results to plot. Hint: model.fit(X) <return>.')
            return None
        if c is None: c=[[0, 0, 0]]
        if (gradient is not None) and ((not isinstance(gradient, str)) or (len(gradient)!=7)): raise Exception('[pca]> Error: gradient must be of type string with Hex color or None.')
        fontdict = _set_fontdict(fontdict)

        # Setup figure
        if fig is None and ax is None:
            # Create entire new figure.
            fig = plt.figure(figsize=figsize)
            if d3:
                ax = fig.add_subplot(projection='3d')
            else:
                ax = fig.add_subplot()
        elif fig is not None and ax is None:
            # Extract axes from fig.
            ax = fig.axes[0]

        # fig, ax = plt.subplots(figsize=figsize, edgecolor='k')
        if fig is not None:
            fig.set_visible(visible)

        # Mark the outliers for plotting purposes.
        Ioutlier1 = np.repeat(False, self.results['PC'].shape[0])
        Ioutlier2 = np.repeat(False, self.results['PC'].shape[0])

        if y is None:
            y, _, _ = self._fig_preprocessing(y, None, d3)

        # Get coordinates
        xs, ys, zs, ax = _get_coordinates(self.results['PC'], PC, fig, ax, d3)

        # Set the markers
        if marker is None: marker='.'
        if isinstance(marker, str): marker = np.repeat(marker, len(xs))
        marker = np.array(marker)
        if len(marker)!=len(xs): raise Exception('Marker length (k=%d) should match the number of samples (n=%d).' %(len(marker), len(xs)))

        # Set Alpha
        if alpha_transparency is None: alpha_transparency=1
        if isinstance(alpha_transparency, (float, int)): alpha_transparency = np.repeat(alpha_transparency, len(xs))
        alpha_transparency = np.array(alpha_transparency)
        if len(alpha_transparency)!=len(xs): raise Exception('alpha_transparency length (k=%d) should match the number of samples (n=%d).' %(len(alpha_transparency), len(xs)))

        # Set Size
        if s is None: s=50
        if isinstance(s, (float, int)): s = np.repeat(s, len(xs))
        s = np.array(s)
        if len(s)!=len(xs): raise Exception('Size (s) length (k=%d) should match the number of samples (n=%d).' %(len(s), len(xs)))

        # Add jitter
        if jitter is not None:
            xs = xs + np.random.normal(0, jitter, size=len(xs))
            if ys is not None: ys = ys + np.random.normal(0, jitter, size=len(ys))
            if zs is not None: zs = zs + np.random.normal(0, jitter, size=len(zs))

        # Get the colors
        if cmap is None:
            # Hide the scatterpoints by making them all white.
            getcolors = np.repeat([1., 1., 1.], len(y), axis=0).reshape(-1, 3)
        else:
            # Figure properties
            xyz, _ = scatterd._preprocessing(xs, ys, zs, y)
            getcolors, fontcolor = scatterd.set_colors(xyz, y, None, c, cmap, gradient=gradient)

        if hotellingt2 and ('y_bool' in self.results['outliers'].columns):
            Ioutlier1 = self.results['outliers']['y_bool'].values
        if SPE and ('y_bool_spe' in self.results['outliers'].columns):
            Ioutlier2 = self.results['outliers']['y_bool_spe'].values
            if not d3:
                # Plot the ellipse
                g_ellipse = spe_dmodx(np.c_[xs, ys], n_std=self.n_std, color='green', calpha=0.3, verbose=0)[1]
                if g_ellipse is not None: ax.add_artist(g_ellipse)

        # Make scatter plot of all not-outliers
        # uiy = np.unique(y)
        # if (len(uiy)==len(y)) and (len(uiy)>=1000) and (label is not None) and np.unique(marker)==1:
        #     if verbose>=2: print('[pca] >Set parameter "label=None" to ignore the labels and significanly speed up the scatter plot.')
        # Add the labels
        # if (label is None):
        #     if d3:
        #         ax.scatter(xs, ys, zs, s=s, alpha=alpha_transparency, color=getcolors, label=None, marker=marker[0])
        #     else:
        #         ax.scatter(xs, ys, s=s, alpha=alpha_transparency, color=getcolors, label=None, marker=marker[0])
        # else:
        for Iloc_sampl, _ in tqdm(enumerate(y), desc="[pca] >Plotting", position=0, leave=False, disable=(verbose==0)):
            if d3:
                ax.scatter(xs[Iloc_sampl], ys[Iloc_sampl], zs[Iloc_sampl], s=np.maximum(s[Iloc_sampl], 0), label=y[Iloc_sampl], alpha=float(alpha_transparency[Iloc_sampl]), color=getcolors[Iloc_sampl, :], marker=marker[Iloc_sampl])
                if label: ax.text(np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]), np.mean(zs[Iloc_sampl]), str(y[Iloc_sampl]), color=[0, 0, 0], fontdict=fontdict)
            else:
                ax.scatter(xs[Iloc_sampl], ys[Iloc_sampl], s=np.maximum(s[Iloc_sampl], 0), label=y[Iloc_sampl], alpha=float(alpha_transparency[Iloc_sampl]), color=getcolors[Iloc_sampl, :], marker=marker[Iloc_sampl])
                if label: ax.text(np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]), str(y[Iloc_sampl]), color=[0, 0, 0], fontdict=fontdict)
                # if label: ax.annotate(yk, np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]))

            # for yk in uiy:
            #     Iloc_sampl = (yk==y)

            #     if d3:
            #         ax.scatter(xs[Iloc_sampl], ys[Iloc_sampl], zs[Iloc_sampl], s=s + 10, label=yk, alpha=alpha_transparency, color=getcolors[Iloc_sampl, :], marker=marker[Iloc_sampl])
            #         if label: ax.text(np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]), np.mean(zs[Iloc_sampl]), str(yk), color=[0, 0, 0], fontdict=fontdict)
            #     else:
            #         ax.scatter(xs[Iloc_sampl], ys[Iloc_sampl], s=s + 10, label=yk, alpha=alpha_transparency, color=getcolors[Iloc_sampl, :], marker=marker[Iloc_sampl])
            #         if label: ax.text(np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]), str(yk), color=[0, 0, 0], fontdict=fontdict)
            #         # if label: ax.annotate(yk, np.mean(xs[Iloc_sampl]), np.mean(ys[Iloc_sampl]))

        # Plot outliers for hotelling T2 test.
        if SPE and ('y_bool_spe' in self.results['outliers'].columns):
            label_spe = str(sum(Ioutlier2)) + ' outliers (SPE/DmodX)'
            if d3:
                ax.scatter(xs[Ioutlier2], ys[Ioutlier2], zs[Ioutlier2], marker='x', color=[0.5, 0.5, 0.5], s=50, label=label_spe, alpha=alpha_transparency)
            else:
                ax.scatter(xs[Ioutlier2], ys[Ioutlier2], marker='d', color=[0.5, 0.5, 0.5], s=50, label=label_spe, alpha=alpha_transparency)
                # Plot the ellipse
                # g_ellipse = spe_dmodx(np.c_[xs, ys], n_std=self.n_std, color='green', calpha=0.3, verbose=0)[1]
                # if g_ellipse is not None: ax.add_artist(g_ellipse)

        # Plot outliers for hotelling T2 test.
        if hotellingt2 and ('y_bool' in self.results['outliers'].columns):
            label_t2 = str(sum(Ioutlier1)) + ' outliers (hotelling t2)'
            if d3:
                ax.scatter(xs[Ioutlier1], ys[Ioutlier1], zs[Ioutlier1], marker='d', color=[0, 0, 0], s=50, label=label_t2, alpha=alpha_transparency)
            else:
                ax.scatter(xs[Ioutlier1], ys[Ioutlier1], marker='x', color=[0, 0, 0], s=50, label=label_t2, alpha=alpha_transparency)

        # Set y
        ax.set_xlabel('PC' + str(PC[0] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[0]] * 100)[0:4] + '% expl.var)')
        if len(self.results['model'].explained_variance_ratio_)>=2:
            ax.set_ylabel('PC' + str(PC[1] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[1]] * 100)[0:4] + '% expl.var)')
        else:
            ax.set_ylabel('PC2 (0% expl.var)')
        if d3 and (len(self.results['model'].explained_variance_ratio_)>=3):
            ax.set_zlabel('PC' + str(PC[2] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[2]] * 100)[0:4] + '% expl.var)')

        if title is None:
            title = str(self.n_components) + ' Principal Components explain [' + str(self.results['pcp'] * 100)[0:5] + '%] of the variance'
        ax.set_title(title)
        if legend: ax.legend()
        ax.grid(True)
        # Return
        return (fig, ax)

    def biplot(self,
               y=None,
               c=None,
               s=50,
               marker='.',
               jitter=None,
               n_feat=None,
               d3=False,
               label=True,
               PC=[0, 1],
               SPE=False,
               hotellingt2=False,
               alpha_transparency=1,
               gradient=None,
               color_arrow='r',
               fontdict={'weight': 'normal', 'size': 12, 'ha': 'center', 'va': 'center', 'c': 'color_arrow'},
               cmap='Set1',
               title=None,
               legend=True,
               figsize=(15, 10),
               visible=True,
               fig=None,
               ax=None,
               verbose=None):
        """Create the Biplot.

        Description
        -----------
        Plots the Principal components with the samples, and the best performing features.
        Per PC, The feature with absolute highest loading is gathered. This can result into features that are seen over multiple PCs, and some features may never be detected.
        For vizualization purposes we will keep only the unique feature-names and plot them with red arrows and green labels.
        The feature-names that were never discovered (described as weak) are colored yellow.

        Parameters
        ----------
        y : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            Color of samples in RGB colors.
            [0,0,0]: If a single color is given, all samples get that color.
        s: Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: '.').
            Marker for the samples.
            '.' : All data points get this marker
            ['.', '*', 's', ..]: Specify per sample the marker type.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        n_feat : int, default: 10
            Number of features that explain the space the most, dervied from the loadings. This parameter is used for vizualization purposes only.
        d3 : Bool, default: False
            3d plot is created when True.
        label : Bool, default: True
            True Show the labels.
            False: Do not show the labels
            None: Ignore all labels (this will significanly speed up the scatterplot)
        PC : list, default : [0, 1]
            Plot the selected Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc.
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
        hotellingt2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
        alpha_transparency: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex (ending) color for the gradient of the scatterplot colors.
            '#FFFFFF'
        color_arrow : String, (default: 'r')
            color for the arrow.
            'r' (default)
        fontdict : dict.
            dictionary containing properties for the arrow font-text.
            Note that the [c]olor: 'color_arrow' inherits the color used in color_arrow.
            {'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': 'color_arrow'}
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : Bool, default: True
            Show the legend based on the unique y-labels.
        figsize : (int, int), optional, default: (15, 10)
            (width, height) in inches.
        visible : Bool, default: True
            Visible status of the Figure. When False, figure is created on the background.
        fig : Figure, optional (default: None)
            Matplotlib figure.
        ax : Axes, optional (default: None)
            Matplotlib Axes object
        Verbose : int (default : 3)
            The higher the number, the more information is printed.
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        tuple containing (fig, ax)

        References
        ----------
            * https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559
            * https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis/50845697#50845697

        """
        if verbose is None: verbose = self.verbose
        if not hasattr(self, 'results'):
            if verbose>=2: print('[pca]> No results to plot. Hint: model.fit(X) <return>.')
            return None

        # Input checks
        fontdict, cmap = _biplot_input_checks(self.results, PC, cmap, fontdict, d3, color_arrow, verbose)

        # Pre-processing
        y, topfeat, n_feat = self._fig_preprocessing(y, n_feat, d3)
        topfeat = pd.concat([topfeat.iloc[PC, :], topfeat.loc[~topfeat.index.isin(PC), :]])
        topfeat.reset_index(inplace=True)

        # Collect coefficients
        coeff = self.results['loadings'].iloc[PC, :]

        # Use the PCs only for scaling purposes
        mean_x = np.mean(self.results['PC'].iloc[:, PC[0]].values)
        mean_y = np.mean(self.results['PC'].iloc[:, PC[1]].values)

        # Plot and scale values for arrows and text by taking the absolute minimum range of the x-axis and y-axis.
        # max_axis = np.min(np.abs(self.results['PC'].iloc[:,0:2]).max())
        max_axis = np.max(np.abs(self.results['PC'].iloc[:, PC]).min(axis=1))
        max_arrow = np.abs(coeff).max().max()
        scale = (np.max([1, np.round(max_axis / max_arrow, 2)])) * 0.93

        # Include additional parameters if 3d-plot is desired.
        if d3:
            if self.results['PC'].shape[1]<3:
                if verbose>=2: print('[pca] >Warning: requires 3 PCs to make 3d plot <return>.')
                return None, None
            mean_z = np.mean(self.results['PC'].iloc[:, PC[2]].values)
            # zs = self.results['PC'].iloc[:,2].values
            fig, ax = self.scatter3d(y=y, label=label, legend=legend, PC=PC, SPE=SPE, hotellingt2=hotellingt2, cmap=cmap, visible=visible, figsize=figsize, alpha_transparency=alpha_transparency, title=title, gradient=gradient, fig=fig, ax=ax, c=c, s=s, jitter=jitter, marker=marker, verbose=verbose)
        else:
            fig, ax = self.scatter(y=y, label=label, legend=legend, PC=PC, SPE=SPE, hotellingt2=hotellingt2, cmap=cmap, visible=visible, figsize=figsize, alpha_transparency=alpha_transparency, title=title, gradient=gradient, fig=fig, ax=ax, c=c, s=s, jitter=jitter, marker=marker, verbose=verbose)

        # For vizualization purposes we will keep only the unique feature-names
        topfeat = topfeat.drop_duplicates(subset=['feature'])
        if topfeat.shape[0]<n_feat:
            n_feat = topfeat.shape[0]
            if verbose>=2: print('[pca] >Warning: n_feat can not be reached because of the limitation of n_components (=%d). n_feat is reduced to %d.' %(self.n_components, n_feat))

        # Plot arrows and text
        texts = []
        for i in range(0, n_feat):
            getfeat = topfeat['feature'].iloc[i]
            label = getfeat + ' (' + ('%.3g' %topfeat['loading'].iloc[i]) + ')'
            getcoef = coeff[getfeat].values
            # Set first PC vs second PC direction. Note that these are not neccarily the best loading.
            xarrow = getcoef[0] * scale  # First PC in the x-axis direction
            yarrow = getcoef[1] * scale  # Second PC in the y-axis direction
            txtcolor = 'y' if topfeat['type'].iloc[i] == 'weak' else 'g'

            if d3:
                zarrow = getcoef[2] * scale
                ax.quiver(mean_x, mean_y, mean_z, xarrow - mean_x, yarrow - mean_y, zarrow - mean_z, color=color_arrow, alpha=0.8, lw=2)
                texts.append(ax.text(xarrow, yarrow, zarrow, label, color=txtcolor, ha='center', va='center'))
            else:
                ax.arrow(mean_x, mean_y, xarrow - mean_x, yarrow - mean_y, color=color_arrow, alpha=0.8, width=0.002, head_width=0.1, head_length=0.1 * 1.1, length_includes_head=True)
                texts.append(ax.text(xarrow, yarrow, label, color=txtcolor, fontdict=fontdict))

        # Plot the adjusted text labels to prevent overlap
        if len(texts)>0: adjust_text(texts)
        # if visible: plt.show()
        return (fig, ax)

    def biplot3d(self,
                 y=None,
                 c=None,
                 s=50,
                 marker='.',
                 jitter=None,
                 n_feat=None,
                 label=True,
                 PC=[0, 1, 2],
                 SPE=False,
                 hotellingt2=False,
                 alpha_transparency=1,
                 gradient=None,
                 color_arrow='r',
                 fontdict={'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': 'color_arrow'},
                 cmap='Set1',
                 title=None,
                 legend=True,
                 figsize=(15, 10),
                 visible=True,
                 fig=None,
                 ax=None,
                 verbose=None):
        """Make biplot in 3d.

        Parameters
        ----------
        y : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            Color of samples in RGB colors.
            [0,0,0]: If a single color is given, all samples get that color.
        s: Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: '.').
            Marker for the samples.
            '.' : All data points get this marker
            ['.', '*', 's', ..]: Specify per sample the marker type.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        n_feat : int, default: 10
            Number of features that explain the space the most, dervied from the loadings. This parameter is used for vizualization purposes only.
        label : Bool, default: True
            True Show the labels.
            False: Do not show the labels
            None: Ignore all labels (this will significanly speed up the scatterplot)
        PC : list, default : [0, 1, 2]
            Plot the selected Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc.
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
        hotellingt2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
        alpha_transparency: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex (ending) color for the gradient of the scatterplot colors.
            '#FFFFFF'
        color_arrow : String, (default: 'r')
            color for the arrow.
            'r' (default)
        fontdict : dict.
            dictionary containing properties for the arrow font-text
            Note that the [c]olor: 'color_arrow' inherits the color used in color_arrow.
            {'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': 'color_arrow'}
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : Bool, default: True
            Show the legend based on the unique y-labels.
        figsize : (int, int), optional, default: (15, 10)
            (width, height) in inches.
        visible : Bool, default: True
            Visible status of the Figure. When False, figure is created on the background.
        fig : Figure, optional (default: None)
            Matplotlib figure.
        ax : Axes, optional (default: None)
            Matplotlib Axes object
        Verbose : int (default : 3)
            The higher the number, the more information is printed.
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        tuple containing (fig, ax)

        """
        if verbose is None: verbose = self.verbose
        if self.results['PC'].shape[1]<3:
            print('[pca] >Requires 3 PCs to make 3d plot. Try to use biplot() instead.')
            return None, None

        fig, ax = self.biplot(y=y,
                              n_feat=n_feat,
                              c=c,
                              s=s,
                              marker=marker,
                              jitter=jitter,
                              d3=True,
                              label=label,
                              PC=PC,
                              SPE=SPE,
                              hotellingt2=hotellingt2,
                              alpha_transparency=alpha_transparency,
                              gradient=gradient,
                              color_arrow=color_arrow,
                              fontdict=fontdict,
                              cmap=cmap,
                              title=title,
                              legend=legend,
                              figsize=figsize,
                              visible=visible,
                              fig=fig,
                              ax=ax,
                              verbose=verbose)

        return (fig, ax)

    # Show explained variance plot
    def plot(self, n_components=None, xsteps=None, title=None, visible=True, figsize=(15, 10), fig=None, ax=None, verbose=None):
        """Scree-plot together with explained variance.

        Parameters
        ----------
        n_components : int [0..1], optional
            Number of PCs that are returned for the plot.
            None: All PCs.
        xsteps : int, optional
            Set the number of xticklabels.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        visible : Bool, default: True
            Visible status of the Figure
            True : Figure is shown.
            False: Figure is created on the background.
        figsize : (int, int)
            (width, height) in inches.
        fig : Figure, optional (default: None)
            Matplotlib figure.
        ax : Axes, optional (default: None)
            Matplotlib Axes object
        Verbose : int (default : 3)
            The higher the number, the more information is printed.
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

        Returns
        -------
        tuple containing (fig, ax)

        """
        if verbose is None: verbose = self.verbose

        if self.method=='sparse_pca':
            print('[pca] >sparse pca does not support variance ratio and therefores scree plots are not supported. <return>')
            return None, None
        if n_components is not None:
            if n_components>len(self.results['explained_var']):
                if verbose>=2: print('[pca] >Warning: Input "n_components=%s" is > then number of PCs (=%s)' %(n_components, len(self.results['explained_var'])))
            n_components = np.minimum(len(self.results['explained_var']), n_components)
            explvarCum = self.results['explained_var'][0:n_components]
            explvar = self.results['variance_ratio'][0:n_components]
        else:
            explvarCum = self.results['explained_var']
            explvar = self.results['variance_ratio']
        xtick_idx = np.arange(1, len(explvar) + 1)

        # Make figure
        if fig is None and ax is None:
            # Create entire new figure.
            fig, ax = plt.subplots(figsize=figsize, edgecolor='k')
        elif fig is not None and ax is None:
            ax = fig.axes[0]

        # Set visibility and plot
        if fig is not None:
            fig.set_visible(visible)
        plt.plot(xtick_idx, explvarCum, 'o-', color='k', linewidth=1, label='Cumulative explained variance')

        # Set xticks if less then 100 datapoints
        if len(explvar)<100:
            ax.set_xticks(xtick_idx)
            xticklabel=xtick_idx.astype(str)
            if xsteps is not None:
                xticklabel[np.arange(1, len(xticklabel), xsteps)] = ''
            ax.set_xticklabels(xticklabel, rotation=90, ha='left', va='top')

        plt.ylabel('Percentage explained variance')
        plt.xlabel('Principle Component')
        plt.ylim([0, 1.05])
        plt.xlim([0, len(explvar) + 1])
        if title is None:
            title = 'Cumulative explained variance\n ' + str(self.n_components) + ' Principal Components explain [' + str(self.results['pcp'] * 100)[0:5] + '%] of the variance.'
        plt.title(title)
        plt.grid(True)

        # Plot vertical line To stress the cut-off point
        ax.axvline(self.n_components, linewidth=0.8, color='r')
        ax.axhline(y=self.results['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
        if len(xtick_idx)<100:
            plt.bar(xtick_idx, explvar, color='#3182bd', alpha=0.8, label='Explained variance')

        if visible:
            plt.show()
            plt.draw()
        # Return
        return (fig, ax)

    # Top scoring components
    def norm(self, X, n_components=None, pcexclude=[1]):
        """Normalize out PCs.

        Description
        -----------
        Normalize your data using the variance seen in hte Principal Components. This allows to remove (technical)
        variation in the data by normalizing out e.g., the 1st or 2nd etc component. This function transforms the
        original data using the PCs that you want to normalize out. As an example, if you aim to remove the variation
        seen in the 1st PC, the returned dataset will contain only the variance seen from the 2nd PC and more.

        Parameters
        ----------
        X : numpy array
            Data set.
        n_components : int [0..1], optional
            Number of PCs that are returned for the plot.
            None: All PCs.
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

        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not isinstance(pcexclude, list): pcexclude=[pcexclude]

        # Fit using PCA
        _ = self.fit_transform(X)
        coeff = self.results['loadings'].values
        score = self.results['PC']
        # Compute explained percentage of variance
        q = self.results['explained_var']
        ndims = np.where(q<=self.n_components)[0]
        ndims = (np.setdiff1d(ndims + 1, pcexclude)) - 1
        # Transform data
        out = np.repeat(np.mean(X, axis=1).reshape(-1, 1), X.shape[1], axis=1) + np.dot(score.values[:, ndims], coeff[:, ndims].T)
        # Return
        return out

    # Import example
    def import_example(self, data='titanic', url=None, sep=','):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep)


# %%
def _get_coordinates(PCs, PC, fig, ax, d3):
    xs = PCs.iloc[:, PC[0]].values
    ys = np.zeros(len(xs))
    zs = None

    # Get y-axis
    if PCs.shape[1]>1:
        ys = PCs.iloc[:, PC[1]].values

    # Get Z-axis
    if d3:
        zs = PCs.iloc[:, PC[2]].values
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    return xs, ys, zs, ax


# %%
def _eigsorted(cov, n_std):
    vals, vecs = np.linalg.eigh(cov)
    # vecs = vecs * np.sqrt(scipy.stats.chi2.ppf(0.95, n_std))
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def spe_dmodx(X, n_std=3, param=None, calpha=0.3, color='green', showfig=False, verbose=3):
    """Compute SPE/distance to model (DmodX).

    Description
    -----------
    Outlier can be detected using SPE/DmodX (distance to model) based on the mean and covariance of the first 2 dimensions of X.
    On the model plane (SPE ≈ 0). Note that the SPE or Hotelling’s T2 are complementary to each other.

    Parameters
    ----------
    X : Array-like
        Input data, in this case the Principal components.
    n_std : int, (default: 3)
        Standard deviation. The default is 2.
    param : 2-element tuple (default: None)
        Pre-computed g_ell_center and cov in the past run. None to compute from scratch with X.
    calpha : float, (default: 0.3)
        transperancy color.
    color : String, (default: 'green')
        Color of the ellipse.
    showfig : bool, (default: False)
        Scatter the points with the ellipse and mark the outliers.

    Returns
    -------
    outliers : pd.DataFrame()
        column with boolean outliers and euclidean distance of each sample to the center of the ellipse.
    ax : object
        Figure axis.
    param : 2-element tuple
        computed g_ell_center and cov from X.
    """
    if verbose>=3: print('[pca] >Outlier detection using SPE/DmodX with n_std=[%d]' %(n_std))
    g_ellipse = None
    # The 2x2 covariance matrix to base the ellipse on the location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
    n_components = np.minimum(2, X.shape[1])
    X = X[:, 0:n_components]

    if X.shape[1]>=2:
        # Compute mean and covariance
        if (param is not None):
            g_ell_center, cov = param
        else:
            g_ell_center = X.mean(axis=0)
            cov = np.cov(X, rowvar=False)
            param = g_ell_center, cov

        # Width and height are "full" widths, not radius
        vals, vecs = _eigsorted(cov, n_std)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        # Compute angles of ellipse
        cos_angle = np.cos(np.radians(180. - angle))
        sin_angle = np.sin(np.radians(180. - angle))
        # Determine the elipse range
        xc = X[:, 0] - g_ell_center[0]
        yc = X[:, 1] - g_ell_center[1]
        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle
        rad_cc = (xct**2 / (width / 2.)**2) + (yct**2 / (height / 2.)**2)

        # Mark the samples outside the ellipse
        outliers = rad_cc>1

        # Plot the raw points.
        g_ellipse = Ellipse(xy=g_ell_center, width=width, height=height, angle=angle, color=color, alpha=calpha)
        y_score = list(map(lambda x: euclidean_distances([g_ell_center], x.reshape(1, -1))[0][0], X))

        if showfig:
            ax = plt.gca()
            ax.add_artist(g_ellipse)
            ax.scatter(X[~outliers, 0], X[~outliers, 1], c='black', linewidths=0.3, label='normal')
            ax.scatter(X[outliers, 0], X[outliers, 1], c='red', linewidths=0.3, label='outlier')
            ax.legend()
    else:
        outliers = np.repeat(False, X.shape[1])
        y_score = np.repeat(None, X.shape[1])

    # Store in dataframe
    out = pd.DataFrame(data={'y_bool_spe': outliers, 'y_score_spe': y_score})
    return out, g_ellipse, param


# %% Outlier detection
def hotellingsT2(X, alpha=0.05, df=1, n_components=5, multipletests='fdr_bh', param=None, verbose=3):
    """Test for outlier using hotelling T2 test.

    Description
    -----------
    Test for outliers using chi-square tests for each of the n_components.
    The resulting P-value matrix is then combined using fishers method per sample.
    The results can be used to priortize outliers as those samples that are an outlier
    across multiple dimensions will be more significant then others.

    Parameters
    ----------
    X : numpy-array.
        Principal Components.
    alpha : float, (default: 0.05)
        Alpha level threshold to determine outliers.
    df : int, (default: 1)
        Degrees of freedom.
    n_components : int, (default: 5)
        Number of PC components to be used to compute the Pvalue.
    multipletests : str, default: 'fdr_bh'
        Multiple testing method. Options are:
            * None : No multiple testing
            * 'bonferroni' : one-step correction
            * 'sidak' : one-step correction
            * 'holm-sidak' : step down method using Sidak adjustments
            * 'holm' : step-down method using Bonferroni adjustments
            * 'simes-hochberg' : step-up method  (independent)
            * 'hommel' : closed method based on Simes tests (non-negative)
            * 'fdr_bh' : Benjamini/Hochberg  (non-negative)
            * 'fdr_by' : Benjamini/Yekutieli (negative)
            * 'fdr_tsbh' : two stage fdr correction (non-negative)
            * 'fdr_tsbky' : two stage fdr correction (non-negative)
    param : 2-element tuple (default: None)
        Pre-computed mean and variance in the past run. None to compute from scratch with X.
    Verbose: int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    Returns
    -------
    outliers : pd.DataFrame
        dataframe containing probability, test-statistics and boolean value.
    y_bools : array-like
        boolean value when significant per PC.
    param : 2-element tuple
        computed mean and variance from X.
    """
    n_components = np.minimum(n_components, X.shape[1])
    X = X[:, 0:n_components]
    y = X

    if (param is not None):
        mean, var = param
    else:
        mean, var = np.mean(X), np.var(X)
        param = (mean, var)
    if verbose>=3: print('[pca] >Outlier detection using Hotelling T2 test with alpha=[%.2f] and n_components=[%d]' %(alpha, n_components))
    y_score = (y - mean) ** 2 / var
    # Compute probability per PC whether datapoints are outside the boundary
    y_proba = 1 - stats.chi2.cdf(y_score, df=df)
    # Set probabilities at a very small value when 0. This is required for the Fishers method. Otherwise inf values will occur.
    y_proba[y_proba==0]=1e-300

    # Compute the anomaly threshold
    anomaly_score_threshold = stats.chi2.ppf(q=(1 - alpha), df=df)
    # Determine for each samples and per principal component the outliers
    y_bools = y_score >= anomaly_score_threshold

    # Combine Pvalues across the components
    Pcomb = []
    # weights = np.arange(0, 1, (1/n_components) )[::-1] + (1/n_components)
    for i in range(0, y_proba.shape[0]):
        # Pcomb.append(stats.combine_pvalues(y_proba[i, :], method='stouffer', weights=weights))
        Pcomb.append(stats.combine_pvalues(y_proba[i, :], method='fisher'))

    Pcomb = np.array(Pcomb)
    # Multiple test correction
    Pcorr = multitest_correction(Pcomb[:, 1], multipletests=multipletests, verbose=verbose)
    # Set dataframe
    outliers = pd.DataFrame(data={'y_proba': Pcorr, 'p_raw': Pcomb[:, 1], 'y_score': Pcomb[:, 0], 'y_bool': Pcorr <= alpha})
    # Return
    return outliers, y_bools, param


# %% Do multiple test correction
def multitest_correction(Praw, multipletests='fdr_bh', verbose=3):
    """Multiple test correction for input pvalues.

    Parameters
    ----------
    Praw : list of float
        Pvalues.
    method : str, default: 'fdr_bh'
        Multiple testing method. Options are:
            * None : No multiple testing
            * 'bonferroni' : one-step correction
            * 'sidak' : one-step correction
            * 'holm-sidak' : step down method using Sidak adjustments
            * 'holm' : step-down method using Bonferroni adjustments
            * 'simes-hochberg' : step-up method  (independent)
            * 'hommel' : closed method based on Simes tests (non-negative)
            * 'fdr_bh' : Benjamini/Hochberg  (non-negative)
            * 'fdr_by' : Benjamini/Yekutieli (negative)
            * 'fdr_tsbh' : two stage fdr correction (non-negative)
            * 'fdr_tsbky' : two stage fdr correction (non-negative)

    Returns
    -------
    list of float.
        Corrected pvalues.

    """
    if multipletests is not None:
        if verbose>=3: print("[pca] >Multiple test correction applied for Hotelling T2 test: [%s]" %(multipletests))
        Padj = multitest.multipletests(Praw, method=multipletests)[1]
    else:
        Padj=Praw

    Padj = np.clip(Padj, 0, 1)
    return Padj


# %% Explained variance
def _explainedvar(X, method='pca', n_components=None, onehot=False, random_state=None, n_jobs=-1, percentExplVar=None, verbose=3):
    # Create the model
    if method=='trunc_svd':
        if verbose>=3: print('[pca] >Fit using Truncated SVD.')
        if n_components is None:
            n_components = X.shape[1] - 1
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif method=='sparse_pca':
        if verbose>=3: print('[pca] >Fit using Sparse PCA.')
        onehot=True
        model = SparsePCA(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
        # model = MiniBatchSparsePCA(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
    else:
        if verbose>=3: print('[pca] >Fit using PCA.')
        model = PCA(n_components=n_components, random_state=random_state)

    # Fit model
    model.fit(X)
    # Do the reduction
    if verbose>=3: print('[pca] >Compute loadings and PCs.')
    loadings = model.components_  # Ook wel de coeeficienten genoemd: coefs!
    PC = model.transform(X)

    # Compute explained variance, top 95% variance
    if (not onehot) and (percentExplVar is None):
        if verbose>=3: print('[pca] >Compute explained variance.')
        percentExplVar = model.explained_variance_ratio_.cumsum()
    # if method=='sparse_pca':
    #     model.explained_variance_ = _get_explained_variance(X.T, PC.T)
    #     model.explained_variance_ratio_ = model.explained_variance_ / model.explained_variance_.sum()
    #     percentExplVar = model.explained_variance_ratio_.cumsum()

    # Return
    return (model, PC, loadings, percentExplVar)


# %% Store results
def _store(PC, loadings, percentExplVar, model_pca, n_components, pcp, col_labels, row_labels, topfeat, outliers, scaler, outliers_params):

    if not outliers.empty: outliers.index = row_labels
    out = {}
    out['loadings'] = loadings
    out['PC'] = pd.DataFrame(data=PC[:, 0:n_components], index=row_labels, columns=loadings.index.values[0:n_components])
    out['explained_var'] = percentExplVar
    if percentExplVar is None:
        out['variance_ratio'] = None
    else:
        out['variance_ratio'] = np.diff(percentExplVar, prepend=0)
    out['model'] = model_pca
    out['scaler'] = scaler
    out['pcp'] = pcp
    out['topfeat'] = topfeat
    out['outliers'] = outliers
    out['outliers_params'] = outliers_params
    return out


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
	verbose : int, (default: 20)
		Print progress to screen. The default is 3.
		60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('Downloading [%s] dataset from github source..' %(data))
        wget.download(url, PATH_TO_DATA)

    # Import local dataset
    if verbose>=3: print('Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %%
def _get_explained_variance(X, components):
    """Get the explained variance.

    Description
    -----------
    Get the explained variance from the principal components of the
    data. This follows the method outlined in [1] section 3.4 (Adjusted Total
    Variance). For an alternate approach (not implemented here), see [2].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The feature vector. n_samples and n_features are the number of
        samples and features, respectively.
    components : array, shape (n_components, n_features)
        The (un-normalized) principle components. [1]

    Notes
    -----
    The variance ratio may not be computed. The main reason is that we
    do not know what the total variance is since we did not compute all
    the components.
    Orthogonality is enforced in this case. Other variants exist that don't
    enforce this [2].

    References
    ----------
        * Journal of Computational and Graphical Statistics, Volume 15, Number 2, Pages 265–286. DOI: 10.1198/106186006X113430.
        * Rodolphe Jenatton, Guillaume Obozinski, Francis Bach ; Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:366-373, 2010.
    """
    # the number of samples
    n_samples = X.shape[0]
    n_components = components.shape[0]
    unit_vecs = components.copy()
    components_norm = np.linalg.norm(components, axis=1)[:, np.newaxis]
    components_norm[components_norm == 0] = 1
    unit_vecs /= components_norm

    # Algorithm, as we compute the adjustd variance for each component, we
    # subtract the variance from components in the direction of previous axes
    proj_corrected_vecs = np.zeros_like(components)
    for i in range(n_components):
        vec = components[i].copy()
        # subtract the previous projections
        for j in range(i):
            vec -= np.dot(unit_vecs[j], vec)*unit_vecs[j]

        proj_corrected_vecs[i] = vec

    # get estimated variance of Y which is matrix product of feature vector
    # and the adjusted components
    Y = np.tensordot(X, proj_corrected_vecs.T, axes=(1, 0))
    YYT = np.tensordot(Y.T, Y, axes=(1, 0))
    explained_variance = np.diag(YYT) / (n_samples - 1)

    return explained_variance


def _biplot_input_checks(results, PC, cmap, fontdict, d3, color_arrow, verbose):
    # Check PCs
    if results['PC'].shape[1]<2: raise ValueError('[pca] >[Error] Requires 2 PCs to make 2d plot.')
    if d3 and len(PC)<3: raise ValueError('[pca] >[Error] in case of biplot3d or d3=True, at least 3 PCs are required.')
    if np.max(PC)>=results['PC'].shape[1]: raise ValueError('[pca] >[Error] PC%.0d does not exist!' %(np.max(PC) + 1))
    if verbose>=3 and d3:
        print('[pca] >Plot PC%.0d vs PC%.0d vs PC%.0d with loadings.' %(PC[0] + 1, PC[1] + 1, PC[2] + 1))
    elif verbose>=3:
        print('[pca] >Plot PC%.0d vs PC%.0d with loadings.' %(PC[0] + 1, PC[1] + 1))
    if cmap is False: cmap=None
    # Set defaults in fontdict
    fontdict =_set_fontdict(fontdict, color_arrow)
    # Set font dictionary
    # Return
    return fontdict, cmap


def _set_fontdict(fontdict, color_arrow=None):
    color_arrow = 'black' if (color_arrow is None) else color_arrow
    fontdict = {**{'weight': 'normal', 'size': 10, 'ha': 'center', 'va': 'center', 'c': color_arrow}, **fontdict}
    if fontdict.get('c')=='color_arrow' and (color_arrow is not None):
        fontdict['c'] = color_arrow
    return fontdict


# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url):
        """Return filename."""
        return os.path.basename(url)

    def download(url, writepath):
        """Download.

        Parameters
        ----------
        url : str.
            Internet source.
        writepath : str.
            Directory to write the file.

        Returns
        -------
        None.

        """
        r = requests.get(url, stream=True)
        with open(writepath, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)