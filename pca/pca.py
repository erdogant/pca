"""pca: A Python Package for Principal Component Analysis."""

import datazets as dz
from scatterd import scatterd
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD  # MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
from matplotlib.patches import Ellipse
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import statsmodels.stats.multitest as multitest
from typing import Union
# import logging

# logger = logging.getLogger('')
# [logger.removeHandler(handler) for handler in logger.handlers[:]]
# console = logging.StreamHandler()
# formatter = logging.Formatter('[pca] >%(levelname)s> %(message)s')
# console.setFormatter(formatter)
# logger.addHandler(console)
# logger = logging.getLogger(__name__)


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

    def __init__(self,
                 n_components=0.95,
                 n_feat=25,
                 method='pca',
                 alpha=0.05,
                 multipletests='fdr_bh',
                 n_std=3,
                 onehot=False,
                 normalize=False,
                 detect_outliers=['ht2', 'spe'],
                 random_state=None,
                 verbose='info'):
        """Initialize pca with user-defined parameters."""
        if isinstance(detect_outliers, str): detect_outliers = [detect_outliers]
        if detect_outliers is not None: detect_outliers=list(map(str.lower, detect_outliers))

        verbose = convert_verbose_to_old(verbose)
        # Convert to new verbose
        # verbose = convert_verbose_to_new(verbose)
        # Set the logger
        # set_logger(verbose=verbose)

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
        >>> fig, ax = model.biplot(SPE=True, HT2=True)
        >>>
        >>> 3D plots
        >>> fig, ax = model.scatter3d()
        >>> fig, ax = model.biplot3d()
        >>> fig, ax = model.biplot3d(SPE=True, HT2=True)
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
            if verbose>=2: print('[pca] >[WARNING]: Sparse or one-hot boolean input data is detected, it is highly recommended to set onehot=True or alternatively, normalize=True')

        # Set col labels
        if isinstance(X, pd.DataFrame) and col_labels is None:
            if verbose>=3: print('[pca] >Extracting column labels from dataframe.')
            col_labels = X.columns.values
        if col_labels is None or len(col_labels)==0 or len(col_labels)!=X.shape[1]:
            if verbose>=3: print('[pca] >Column labels are auto-completed.')
            col_labels = np.arange(1, X.shape[1] + 1).astype(str)
        # if isinstance(col_labels, list):
        col_labels=np.array(col_labels)

        # Set row labels
        if isinstance(X, pd.DataFrame) and row_labels is None:
            if verbose>=3: print('[pca] >Extracting row labels from dataframe.')
            row_labels = X.index.values
        if row_labels is None or len(row_labels)!=X.shape[0]:
            # row_labels = np.ones(X.shape[0]).astype(int)
            row_labels = np.arange(0, X.shape[0]).astype(int)
            if verbose>=3: print('[pca] >Row labels are auto-completed.')
        # if isinstance(row_labels, list):
        row_labels=np.array(row_labels)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if sp.issparse(X) and self.normalize:
            if verbose>=3: print('[pca] >[WARNING]: Can not normalize a sparse matrix. Normalize is set to [False]')
            self.normalize=False
        if (sp.issparse(X) is False) and (self.n_components > X.shape[1]):
            # raise Exception('[pca] >Number of components can not be more then number of features.')
            if verbose>=2: print('[pca] >[WARNING]: >Number of components can not be more then number of features. n_components is set to %d' %(X.shape[1] - 1))
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

        return X, row_labels, col_labels, scaler

    # Figure pre processing
    def _fig_preprocessing(self, labels, n_feat, d3):
        if hasattr(self, 'PC'): raise Exception('[pca] >Error: Principal components are not derived yet. Tip: run fit_transform() first.')
        if self.results['PC'].shape[1]<1: raise Exception('[pca] >Requires at least 1 PC to make plot.')

        if (n_feat is not None):
            topfeat = self.compute_topfeat()
            # n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[0]), 2)
        else:
            topfeat = self.results['topfeat']
            n_feat = self.n_feat

        if d3:
            n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[1]), 1)
        else:
            n_feat = np.maximum(np.minimum(n_feat, self.results['loadings'].shape[1]), 1)

        if (labels is not None):
            if len(labels)!=self.results['PC'].shape[0]: raise Exception('[pca] >Error: Input variable [labels] should have some length as the number input samples: [%d].' %(self.results['PC'].shape[0]))
            labels = labels.astype(str)
        else:
            labels = self.results['PC'].index.values.astype(str)

        # if all labels appear to be not uniuqe. Do not plot because it takes to much time.
        if len(np.unique(labels))==self.results['PC'].shape[0]: labels=None

        if self.method=='sparse_pca':
            print('[pca] >sparse pca does not supported variance ratio and therefore, biplots will not be supported. <return>')
            self.results['explained_var'] = [None, None]
            self.results['model'].explained_variance_ratio_ = [0, 0]
            self.results['pcp'] = 0

        if (self.results['explained_var'] is None) or len(self.results['explained_var'])<=1:
            raise Exception('[pca] >Error: No PCs are found with explained variance.')

        return labels, topfeat, n_feat

    # Scatter plot
    def scatter(self,
                labels=None,
                c=[0, 0.1, 0.4],
                s=150,
                marker='o',
                edgecolor='#000000',
                SPE=False,
                HT2=False,
                jitter=None,
                PC=None,
                alpha=0.8,
                gradient=None,
                density=False,
                density_on_top=False,
                fontcolor=[0, 0, 0],
                fontsize=18,
                fontweight='normal',
                cmap='tab20c',
                title=None,
                legend=None,
                figsize=(25, 15),
                dpi=100,
                visible=True,
                fig=None,
                ax=None,
                grid=True,
                y=None,  # deprecated
                label=None,  # deprecated
                verbose=3):
        """Scatter 2d plot.

        Parameters
        ----------
        labels : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            The marker colors. Possible values:
                * A scalar or sequence of n numbers to be mapped to colors using cmap and norm.
                * A 2D array in which the rows are RGB or RGBA.
                * A sequence of colors of length n.
                * A single color format string.
        s: Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: 'o').
            Marker for the samples.
                * 'x' : All data points get this marker
                * ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X') : Specify per sample the marker type.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        PC : tupel, default: None
            Plot the selected Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc.
            None : Take automatically the first 2 components and 3 in case d3=True.
            [0, 1] : Define the PC for 2D.
            [0, 1, 2] : Define the PCs for 3D
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
                * None : Auto detect. If outliers are detected. it is set to True.
                * True : Show outliers
                * False : Do not show outliers
        HT2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
                * None : Auto detect. If outliers are detected. it is set to True.
                * True : Show outliers
                * False : Do not show outliers
        alpha: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex color to make a lineair gradient for the scatterplot.
            '#FFFFFF'
        density : Bool (default: False)
            Include the kernel density in the scatter plot.
        density_on_top : bool, (default: False)
            True : The density is the highest layer.
            False : The density is the lowest layer.
        fontsize : String, optional
            The fontsize of the y labels that are plotted in the graph. The default is 16.
        fontcolor: list/array of RGB colors with same size as X (default : None)
            None : Use same colorscheme as for c
            '#000000' : If the input is a single color, all fonts will get this color.
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : int, default: None
            None: Set automatically based on number of labels.
            False : No legend.
            True : Best position.
            1 : 'upper right'
            2 : 'upper left'
            3 : 'lower left'
            4 : 'lower right'
        figsize : (int, int), optional, default: (25, 15)
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
        _show_deprecated_warning(label, y, verbose)
        if not hasattr(self, 'results'):
            if verbose>=2: print('[pca]> No results to plot. Hint: model.fit(X) <return>.')
            return None, None
        if (PC is not None) and self.results['PC'].shape[1]<len(PC):
            if verbose>=1: print('[pca]> 3D plot requires 3 PCs <return>.')
            return None, None

        # Set parameters based on intuition
        if c is None: s=0
        if cmap is None: s=0
        if alpha is None: alpha=0.8
        if PC is None: PC=[0, 1]
        d3 = True if len(PC)==3 else False

        # Set the labels
        if labels is None: labels, _, _ = self._fig_preprocessing(labels, None, d3)
        # Get coordinates
        xs, ys, zs = _get_coordinates(self.results['PC'], PC)
        # Setup figure
        # fig, ax = _setup_figure(fig, ax, d3, visible, figsize, dpi)

        fig, ax = scatterd(x=xs,
                           y=ys,
                           z=zs,
                           s=s,
                           c=c,
                           labels=labels,
                           edgecolor=edgecolor,
                           alpha=alpha,
                           marker=marker,
                           jitter=jitter,
                           density=density,
                           opaque_type='per_class',
                           density_on_top=density_on_top,
                           gradient=gradient,
                           cmap=cmap,
                           legend=legend,
                           fontcolor=fontcolor,
                           fontsize=fontsize,
                           fontweight=fontweight,
                           grid=grid,
                           dpi=dpi,
                           figsize=figsize,
                           visible=visible,
                           fig=fig,
                           ax=ax)

        # Plot the SPE with Elipse
        fig, ax = _add_plot_SPE(self, xs, ys, zs, SPE, d3, alpha, s, fig, ax)
        # Plot hotelling T2
        fig, ax = _add_plot_HT2(self, xs, ys, zs, HT2, d3, alpha, s, fig, ax)
        # Add figure properties
        fig, ax = _add_plot_properties(self, PC, d3, title, legend, labels, fig, ax, fontsize, verbose)
        # Return
        return (fig, ax)

    def biplot(self,
               labels=None,
               c=[0, 0.1, 0.4],
               s=150,
               marker='o',
               edgecolor='#000000',
               jitter=None,
               n_feat=None,
               PC=None,
               SPE=None,
               HT2=None,
               alpha=0.8,
               gradient=None,
               density=False,
               density_on_top=False,
               fontcolor=[0, 0, 0],
               fontsize=18,
               fontweight='normal',
               color_arrow=None,
               arrowdict={'fontsize': None, 'color_text': None, 'weight': None, 'alpha': None, 'color_strong': '#880808', 'color_weak': '#002a77', 'scale_factor': None},
               cmap='tab20c',
               title=None,
               legend=None,
               figsize=(25, 15),
               visible=True,
               fig=None,
               ax=None,
               dpi=100,
               grid=True,
               y=None,  # deprecated
               label=None,  # deprecated
               verbose=None):
        """Create Biplot.

        Plots the Principal components with the samples, and the best performing features.
        Per PC, The feature with absolute highest loading is gathered. This can result into features that are seen over multiple PCs, and some features may never be detected.
        For vizualization purposes we will keep only the unique feature-names and plot them with red arrows and green labels.
        The feature-names that were never discovered (described as weak) are colored yellow.

        Parameters
        ----------
        labels : array-like, default: None
            Label for each sample. The labeling is used for coloring the samples.
        c: list/array of RGB colors for each sample.
            The marker colors. Possible values:
                * A scalar or sequence of n numbers to be mapped to colors using cmap and norm.
                * A 2D array in which the rows are RGB or RGBA.
                * A sequence of colors of length n.
                * A single color format string.
        s : Int or list/array (default: 50)
            Size(s) of the scatter-points.
            [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
            50: all points get this size.
        marker: list/array of strings (default: 'o').
            Marker for the samples.
                * 'x' : All data points get this marker
                * ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X') : Specify per sample the marker type.
        n_feat : int, default: 10
            Number of features that explain the space the most, dervied from the loadings. This parameter is used for vizualization purposes only.
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        PC : tupel, default: None
            Plot the selected Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc.
            None : Take automatically the first 2 components and 3 in case d3=True.
            [0, 1] : Define the PC for 2D.
            [0, 1, 2] : Define the PCs for 3D
        SPE : Bool, default: False
            Show the outliers based on SPE/DmodX method.
                * None : Auto detect. If outliers are detected. it is set to True.
                * True : Show outliers
                * False : Do not show outliers
        HT2 : Bool, default: False
            Show the outliers based on the hotelling T2 test.
                * None : Auto detect. If outliers are detected. it is set to True.
                * True : Show outliers
                * False : Do not show outliers
        alpha: float or array-like of floats (default: 1).
            The alpha blending value ranges between 0 (transparent) and 1 (opaque).
            1: All data points get this alpha
            [1, 0.8, 0.2, ...]: Specify per sample the alpha
        gradient : String, (default: None)
            Hex (ending) color for the gradient of the scatterplot colors.
            '#FFFFFF'
        density : Bool (default: False)
            Include the kernel density in the scatter plot.
        density_on_top : bool, (default: False)
            True : The density is the highest layer.
            False : The density is the lowest layer.
        fontsize : String (default: 16)
            The fontsize of the labels.
        fontcolor: list/array of RGB colors with same size as X (default : None)
            None : Use same colorscheme as for c
            '#000000' : If the input is a single color, all fonts will get this color.
        fontweight : String, (default: 'normal')
            The fontweight of the labels.
                * 'normal'
                * 'bold'
        color_arrow : String, (default: None)
            color for the arrow.
                * None: Color arrows based on strength using 'color_strong' and 'color_weak'.
                * '#000000'
                * 'r'
        arrowdict : dict.
            Dictionary containing properties for the arrow font-text.
            {'fontsize': None, 'color_text': None, 'weight': None, 'alpha': None, 'color_strong': '#880808', 'color_weak': '#002a77', 'ha': 'center', 'va': 'center'}
                * fontsize: None automatically adjust based on the fontsize. Specify otherwise.
                * 'color_text': None automatically adjust color based color_strong and color_weak. Specify hex color otherwise.
                * 'weight': None automatically adjust based on fontweight. Specify otherwise: 'normal', 'bold'
                * 'alpha': None automatically adjust based on loading value.
                * 'color_strong': Hex color for strong loadings (color_arrow needs to be set to None).
                * 'color_weak': Hex color for weak loadings (color_arrow needs to be set to None).
                * 'scale_factor': The scale factor for the arrow length. None automatically sets changes between 2d and 3d plots.
        cmap : String, optional, default: 'Set1'
            Colormap. If set to None, no points are shown.
        title : str, default: None
            Title of the figure.
            None: Automatically create title text based on results.
            '' : Remove all title text.
            'title text' : Add custom title text.
        legend : int, default: None
            None: Set automatically based on number of labels.
            False : No legend.
            True : Best position.
            1 : 'upper right'
            2 : 'upper left'
            3 : 'lower left'
            4 : 'lower right'
        figsize : (int, int), optional, default: (25, 15)
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
        _show_deprecated_warning(label, y, verbose)
        if not hasattr(self, 'results'):
            if verbose>=2: print('[pca]> [WARNING]: No results to plot. Hint: model.fit(X) <return>')
            return None, None

        # Input checks
        arrowdict, cmap, PC, d3, s = _biplot_input_checks(self.results, PC, cmap, arrowdict, color_arrow, fontsize, fontweight, c, s, verbose)
        # Pre-processing
        labels, topfeat, n_feat = self._fig_preprocessing(labels, n_feat, d3)
        # Scatterplot
        fig, ax = self.scatter(labels=labels, legend=legend, PC=PC, SPE=SPE, HT2=HT2, cmap=cmap, visible=visible, figsize=figsize, alpha=alpha, title=title, gradient=gradient, fig=fig, ax=ax, c=c, s=s, jitter=jitter, marker=marker, fontcolor=fontcolor, fontweight=fontweight, fontsize=fontsize, edgecolor=edgecolor, density=density, density_on_top=density_on_top, dpi=dpi, grid=grid, verbose=verbose)
        # Add the loadings with arrow to the plot
        fig, ax = _plot_loadings(self, topfeat, n_feat, PC, d3, arrowdict, fig, ax, verbose)
        # Plot
        # if visible: plt.show()
        # Return
        return fig, ax

    def scatter3d(self, PC=[0, 1, 2], **args):
        """Scatter 3d plot.

        Parameters
        ----------
        Input parameters are described under <scatter>.
        """
        fig, ax = self.scatter(PC=PC, **args)
        return fig, ax

    def biplot3d(self, PC=[0, 1, 2], alpha=0.8, figsize=(30, 25), **args):
        """Biplot 3d plot.

        Parameters
        ----------
        Input parameters are described under <scatter>.
        """
        if not isinstance(alpha, (int, float)): alpha=0.8
        fig, ax = self.biplot(PC=PC, alpha=alpha, figsize=figsize, **args)
        return fig, ax

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
        figsize : (int, int): (default: 25, 15)
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
                if verbose>=2: print('[pca] >[WARNING]: Input "n_components=%s" is > then number of PCs (=%s)' %(n_components, len(self.results['explained_var'])))
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
        if fig is not None: fig.set_visible(visible)
        plt.plot(xtick_idx, explvarCum, 'o-', color='k', linewidth=1, label='Cumulative explained variance', visible=visible)

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

        # if visible:
        #     plt.draw()
        #     plt.show()
        # Return
        return (fig, ax)

    # Top scoring components
    def norm(self, X, n_components=None, pcexclude=[1]):
        """Normalize out PCs.

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
    def import_example(self, data='iris', url=None, sep=','):
        """Import example dataset from github source.

        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets
                * 'iris'
                * 'sprinkler'
                * 'titanic'
                * 'student'
                * 'fifa'
                * 'cancer'
                * 'waterpump'
                * 'retail'
        url : str
            url link to to dataset.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep, verbose=0)


# %%
def _get_coordinates(PCs, PC):
    xs = PCs.iloc[:, PC[0]].values
    ys = np.zeros(len(xs))
    zs = None

    # Get y-axis
    if PCs.shape[1]>1:
        ys = PCs.iloc[:, PC[1]].values

    # Get Z-axis
    if len(PC)>=3:
        zs = PCs.iloc[:, PC[2]].values

    return xs, ys, zs


# %%
def _eigsorted(cov, n_std):
    vals, vecs = np.linalg.eigh(cov)
    # vecs = vecs * np.sqrt(scipy.stats.chi2.ppf(0.95, n_std))
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def spe_dmodx(X, n_std=3, param=None, calpha=0.3, color='green', visible=False, verbose=3):
    """Compute SPE/distance to model (DmodX).

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
    visible : bool, (default: False)
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

        if visible:
            ax = plt.gca()
            ax.add_artist(g_ellipse)
            ax.scatter(X[~outliers, 0], X[~outliers, 1], c='black', linewidths=0.3, label='normal')
            ax.scatter(X[outliers, 0], X[outliers, 1], c='red', linewidths=0.3, label='outlier')
            ax.legend(loc=0)
    else:
        outliers = np.repeat(False, X.shape[1])
        y_score = np.repeat(None, X.shape[1])

    # Store in dataframe
    out = pd.DataFrame(data={'y_bool_spe': outliers, 'y_score_spe': y_score})
    return out, g_ellipse, param


# %% Outlier detection
def hotellingsT2(X, alpha=0.05, df=1, n_components=5, multipletests='fdr_bh', param=None, verbose=3):
    """Test for outlier using hotelling T2 test.

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
def import_example(data='iris', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets
            * 'iris'
            * 'sprinkler'
            * 'titanic'
            * 'student'
            * 'fifa'
            * 'cancer'
            * 'waterpump'
            * 'retail'
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
    return dz.get(data, url=url, sep=sep, verbose=0)


# %%
def _get_explained_variance(X, components):
    """Get the explained variance.

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
            vec -= np.dot(unit_vecs[j], vec) * unit_vecs[j]

        proj_corrected_vecs[i] = vec

    # get estimated variance of Y which is matrix product of feature vector
    # and the adjusted components
    Y = np.tensordot(X, proj_corrected_vecs.T, axes=(1, 0))
    YYT = np.tensordot(Y.T, Y, axes=(1, 0))
    explained_variance = np.diag(YYT) / (n_samples - 1)

    return explained_variance


def _biplot_input_checks(results, PC, cmap, arrowdict, color_arrow, fontsize, fontweight, c, s, verbose):
    if c is None: s=0
    if cmap is None: s=0
    if isinstance(s, (int, float)) and s==0: fontsize=0
    if PC is None: PC=[0, 1]
    d3 = True if len(PC)>=3 else False

    # Check PCs
    if results['PC'].shape[1]<2: raise ValueError('[pca] >[Error] Requires 2 PCs to make 2d plot.')
    if np.max(PC)>=results['PC'].shape[1]: raise ValueError('[pca] >[Error] PC%.0d does not exist!' %(np.max(PC) + 1))
    if verbose>=3 and d3:
        print('[pca] >Plot PC%.0d vs PC%.0d vs PC%.0d with loadings.' %(PC[0] + 1, PC[1] + 1, PC[2] + 1))
    elif verbose>=3:
        print('[pca] >Plot PC%.0d vs PC%.0d with loadings.' %(PC[0] + 1, PC[1] + 1))

    # Set defaults in arrowdict
    arrowdict =_set_arrowdict(arrowdict, color_arrow=color_arrow, fontsize=fontsize, fontweight=fontweight)
    # Return
    return arrowdict, cmap, PC, d3, s


def _set_arrowdict(arrowdict, color_arrow=None, fontsize=18, fontweight='normal'):
    # color_arrow = None if (color_arrow is None) else color_arrow
    arrowdict = {**{'color_arrow': color_arrow, 'color_text': None, 'fontsize': 18 if fontsize==0 else fontsize, 'weight': fontweight, 'alpha': None, 'ha': 'center', 'va': 'center', 'color_strong': '#880808', 'color_weak': '#002a77', 'scale_factor': None}, **arrowdict}
    if arrowdict.get('fontsize') is None:
        arrowdict['fontsize'] = 18 if fontsize==0 else fontsize
    if arrowdict.get('weight') is None:
        arrowdict['weight'] = fontweight
    return arrowdict


def _plot_loadings(self, topfeat, n_feat, PC, d3, arrowdict, fig, ax, verbose):
    topfeat = pd.concat([topfeat.iloc[PC, :], topfeat.loc[~topfeat.index.isin(PC), :]])
    topfeat.reset_index(inplace=True)
    # Collect coefficients
    coeff = self.results['loadings'].iloc[PC, :]

    # Use the PCs only for scaling purposes
    mean_x = np.mean(self.results['PC'].iloc[:, PC[0]].values)
    mean_y = np.mean(self.results['PC'].iloc[:, PC[1]].values)
    if d3: mean_z = np.mean(self.results['PC'].iloc[:, PC[2]].values)

    # Plot and scale values for arrows and text by taking the absolute minimum range of the x-axis and y-axis.
    max_axis = np.max(np.abs(self.results['PC'].iloc[:, PC]).min(axis=1))
    max_arrow = np.abs(coeff).max().max()
    if arrowdict['scale_factor'] is None:
        scale_factor = 1.8 if d3 else 1.1
    else:
        scale_factor = arrowdict['scale_factor']
    # Scale the arrows using the scale factor
    scale = (np.max([1, np.round(max_axis / max_arrow, 2)])) * scale_factor

    # For vizualization purposes we will keep only the unique feature-names
    topfeat = topfeat.drop_duplicates(subset=['feature'])
    if topfeat.shape[0]<n_feat:
        n_feat = topfeat.shape[0]
        if verbose>=2: print('[pca] >[WARNING]: n_feat can not be reached because of the limitation of n_components (=%d). n_feat is reduced to %d.' %(self.n_components, n_feat))

    # Plot arrows and text
    arrow_line_color = arrowdict['color_arrow']
    arrow_text_color = arrowdict['color_text']
    arrow_alpha = arrowdict['alpha']
    alpha_scaled = normalize_size(topfeat['loading'].abs().values.reshape(-1, 1), minscale=0.35, maxscale=0.95, scaler='minmax')
    texts = []
    for i in range(0, n_feat):
        getfeat = topfeat['feature'].iloc[i]
        label = getfeat + ' (' + ('%.3g' %topfeat['loading'].iloc[i]) + ')'
        getcoef = coeff[getfeat].values
        # Set first PC vs second PC direction. Note that these are not neccarily the best loading.
        xarrow = getcoef[0] * scale  # First PC in the x-axis direction
        yarrow = getcoef[1] * scale  # Second PC in the y-axis direction
        # Set the arrow and arrow-text colors
        # Update arrow color if None
        loading_color = arrowdict['color_weak'] if topfeat['type'].iloc[i] == 'weak' else arrowdict['color_strong']
        # Update colors if None
        if arrowdict['color_arrow'] is None: arrow_line_color = loading_color
        if arrowdict['color_text'] is None: arrow_text_color = loading_color
        if arrowdict['alpha'] is None: arrow_alpha = alpha_scaled[i]

        if d3:
            zarrow = getcoef[2] * scale
            ax.quiver(mean_x, mean_y, mean_z, xarrow - mean_x, yarrow - mean_y, zarrow - mean_z, color=arrow_line_color, alpha=arrow_alpha, lw=2)
            texts.append(ax.text(xarrow, yarrow, zarrow, label, fontsize=arrowdict['fontsize'], c=arrow_text_color, weight=arrowdict['weight'], ha=arrowdict['ha'], va=arrowdict['va'], zorder=25))
        else:
            head_width = 0.1
            ax.arrow(mean_x, mean_y, xarrow - mean_x, yarrow - mean_y, color=arrow_line_color, alpha=arrow_alpha, width=0.002, head_width=head_width, head_length=head_width * 1.1, length_includes_head=True, zorder=10)
            texts.append(ax.text(xarrow, yarrow, label, fontsize=arrowdict['fontsize'], c=arrow_text_color, weight=arrowdict['weight'], ha=arrowdict['ha'], va=arrowdict['va'], zorder=10))

    # Plot the adjusted text labels to prevent overlap. Do not adjust text in 3d plots as it will mess up the locations.
    if len(texts)>0 and not d3: adjust_text(texts)

    # Return
    return fig, ax


def _add_plot_SPE(self, xs, ys, zs, SPE, d3, alpha, s, fig, ax):
    # Get the outliers
    Ioutlier2 = np.repeat(False, self.results['PC'].shape[0])
    if SPE and ('y_bool_spe' in self.results['outliers'].columns):
        Ioutlier2 = self.results['outliers']['y_bool_spe'].values
        if not d3:
            # Plot the ellipse
            g_ellipse = spe_dmodx(np.c_[xs, ys], n_std=self.n_std, color='green', calpha=0.1, visible=False, verbose=0)[1]
            if g_ellipse is not None:
                ax.add_artist(g_ellipse)
                # Set the order of the layer at 1. At this point it is over the density layer which looks nicer.
                g_ellipse.set_zorder(1)

    # Plot outliers for hotelling T2 test.
    if isinstance(s, (int, float)): s = 150 if s>0 else 0
    if not isinstance(s, (int, float)): s=150
    if SPE and ('y_bool_spe' in self.results['outliers'].columns):
        label_spe = str(sum(Ioutlier2)) + ' outlier(s) (SPE/DmodX)'
        if d3:
            ax.scatter(xs[Ioutlier2], ys[Ioutlier2], zs[Ioutlier2], marker='x', color=[0.5, 0.5, 0.5], s=s, label=label_spe, alpha=alpha, zorder=15)
        else:
            ax.scatter(xs[Ioutlier2], ys[Ioutlier2], marker='d', color=[0.5, 0.5, 0.5], s=s, label=label_spe, alpha=alpha, zorder=15)

    return fig, ax


def _add_plot_HT2(self, xs, ys, zs, HT2, d3, alpha, s, fig, ax):
    # Plot outliers for hotelling T2 test.
    if isinstance(s, (int, float)): s = 150 if s>0 else 0
    if not isinstance(s, (int, float)): s=150
    # Plot outliers for hotelling T2 test.
    if HT2 and ('y_bool' in self.results['outliers'].columns):
        Ioutlier1 = self.results['outliers']['y_bool'].values

    if HT2 and ('y_bool' in self.results['outliers'].columns):
        label_t2 = str(sum(Ioutlier1)) + ' outlier(s) (hotelling t2)'
        if d3:
            ax.scatter(xs[Ioutlier1], ys[Ioutlier1], zs[Ioutlier1], marker='d', color=[0, 0, 0], s=s, label=label_t2, alpha=alpha, zorder=15)
        else:
            ax.scatter(xs[Ioutlier1], ys[Ioutlier1], marker='x', color=[0, 0, 0], s=s, label=label_t2, alpha=alpha, zorder=15)

    return fig, ax


def _add_plot_properties(self, PC, d3, title, legend, labels, fig, ax, fontsize, verbose):
    # Set labels
    ax.set_xlabel('PC' + str(PC[0] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[0]] * 100)[0:4] + '% expl.var)')
    if len(self.results['model'].explained_variance_ratio_)>=2:
        ax.set_ylabel('PC' + str(PC[1] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[1]] * 100)[0:4] + '% expl.var)')
    else:
        ax.set_ylabel('PC2 (0% expl.var)')
    if d3 and (len(self.results['model'].explained_variance_ratio_)>=3):
        ax.set_zlabel('PC' + str(PC[2] + 1) + ' (' + str(self.results['model'].explained_variance_ratio_[PC[2]] * 100)[0:4] + '% expl.var)')

    # Set title
    if title is None:
        title = str(self.n_components) + ' Principal Components explain [' + str(self.results['pcp'] * 100)[0:5] + '%] of the variance'

    # Determine the legend status if set to None
    if isinstance(legend, bool): legend = 0 if legend else -1
    if legend is None: legend = -1 if len(np.unique(labels))>20 else 0
    if legend>=0: ax.legend(loc=legend, fontsize=14)

    ax.set_title(title, fontsize=18)
    ax.grid(True)
    # Return
    return fig, ax


def normalize_size(getsizes, minscale: Union[int, float] = 0.5, maxscale: Union[int, float] = 4, scaler: str = 'zscore'):
    """Normalize values between minimum and maximum value.

    Parameters
    ----------
    getsizes : input array
        Array of values that needs to be scaled.
    minscale : Union[int, float], optional
        Minimum value. The default is 0.5.
    maxscale : Union[int, float], optional
        Maximum value. The default is 4.
    scaler : str, optional
        Type of scaler. The default is 'zscore'.
            * 'zscore'
            * 'minmax'

    Returns
    -------
    getsizes : array-like
        scaled values between min-max.

    """
    # Instead of Min-Max scaling, that shrinks any distribution in the [0, 1] interval, scaling the variables to
    # Z-scores is better. Min-Max Scaling is too sensitive to outlier observations and generates unseen problems,

    # Set sizes to 0 if not available
    getsizes[np.isinf(getsizes)]=0
    getsizes[np.isnan(getsizes)]=0

    # out-of-scale datapoints.
    if scaler == 'zscore' and len(np.unique(getsizes)) > 3:
        getsizes = (getsizes.flatten() - np.mean(getsizes)) / np.std(getsizes)
        getsizes = getsizes + (minscale - np.min(getsizes))
    elif scaler == 'minmax':
        try:
            from sklearn.preprocessing import MinMaxScaler
        except:
            raise Exception('sklearn needs to be pip installed first. Try: pip install scikit-learn')
        # scaling
        getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    else:
        getsizes = getsizes.ravel()
    # Max digits is 4
    getsizes = np.array(list(map(lambda x: round(x, 4), getsizes)))

    return getsizes


# %%
def convert_verbose_to_old(verbose):
    """Convert new verbosity to the old ones."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if isinstance(verbose, str) or verbose>=10:
        status_map = {
            60: 0, 'silent': 0, 'off': 0, 'no': 0, None: 0,
            40: 1, 'error': 1, 'critical': 1,
            30: 2, 'warning': 2,
            20: 3, 'info': 3,
            10: 4, 'debug': 4}
        return status_map.get(verbose, 0)
    else:
        return verbose


# def convert_verbose_to_new(verbose):
#     """Convert old verbosity to the new."""
#     # In case the new verbosity is used, convert to the old one.
#     if verbose is None: verbose=0
#     if not isinstance(verbose, str) and verbose<10:
#         status_map = {
#             'None': 'silent',
#             0: 'silent',
#             6: 'silent',
#             1: 'critical',
#             2: 'warning',
#             3: 'info',
#             4: 'debug',
#             5: 'debug'}
#         if verbose>=2: print('[scatterd]> WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
#         return status_map.get(verbose, 0)
#     else:
#         return verbose


# def get_logger():
#     return logger.getEffectiveLevel()


# def set_logger(verbose: [str, int] = 'info'):
#     """Set the logger for verbosity messages.

#     Parameters
#     ----------
#     verbose : [str, int], default is 'info' or 20
#         Set the verbose messages using string or integer values.
#         * [0, 60, None, 'silent', 'off', 'no']: No message.
#         * [10, 'debug']: Messages from debug level and higher.
#         * [20, 'info']: Messages from info level and higher.
#         * [30, 'warning']: Messages from warning level and higher.
#         * [50, 'critical']: Messages from critical level and higher.

#     Returns
#     -------
#     None.

#     > # Set the logger to warning
#     > set_logger(verbose='warning')
#     > # Test with different messages
#     > logger.debug("Hello debug")
#     > logger.info("Hello info")
#     > logger.warning("Hello warning")
#     > logger.critical("Hello critical")

#     """
#     # Set 0 and None as no messages.
#     if (verbose==0) or (verbose is None):
#         verbose=60
#     # Convert to verbose
#     verbose = convert_verbose_to_new(verbose)
#     # Convert str to levels
#     if isinstance(verbose, str):
#         levels = {'silent': 60,
#                   'off': 60,
#                   'no': 60,
#                   'debug': 10,
#                   'info': 20,
#                   'warning': 30,
#                   'error': 50,
#                   'critical': 50}
#         verbose = levels[verbose]

#     # Show examples
#     logger.setLevel(verbose)
#     logger.debug('Set verbose to %s' %(verbose))


# def disable_tqdm():
#     """Set the logger for verbosity messages."""
#     return (True if (logger.getEffectiveLevel()>=30) else False)


# %%
def _show_deprecated_warning(label, y, verbose):
    if label is not None:
        if verbose>=2: print('[pca]> [WARNING]: De parameter <label> is deprecated and will not be supported in future version.')
    if y is not None:
        if verbose>=2: print('[pca]> [WARNING]: De parameter <y> is deprecated and will not be supported in future version. Use <labels> instead.')
