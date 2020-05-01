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


# %% Make PCA fit
def fit(X, n_components=None, sparse_data=False, row_labels=[], col_labels=[], random_state=None, normalize=True, verbose=3):
    """Fit PCA on data.

    Parameters
    ----------
    X : numpy array
        [NxM] array with columns as features and rows as samples.
    sparse_data : [Bool] optional, default=False
        Boolean: Set True if X is a sparse data set such as the output of a tfidf model. Many zeros and few numbers. Note this is different then a sparse matrix. Sparse data can be in a sparse matrix.
    n_components : [0,..,1] or [1,..number of samples-1] optional
        Number of TOP components to be returned. Values>0 are the number of components. Values<0 are the components that covers at least the percentage of variance.
        0.95: (default) Take the number of components that cover at least 95% of variance
        2:    Take the top 2 components
        None: All
    row_labels : [list of integers or strings] optional
        Used for colors
    col_labels : [list of string] optional
        Numpy Vector of strings: Name of the features that represent the data features and loadings
    random_state : int optional
        Random state
    normalize : bool optional
        Normalize data, Z-score (default True)
    
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
    >>> X = load_iris().data
    >>> labels = load_iris().feature_names
    >>> y = load_iris().target
    >>> Fit using PCA
    >>> model = pca.fit(X, row_labels=y, col_labels=labels)
    >>> ax = pca.biplot(model) 
    >>> ax = pca.biplot3d(model)
    >>> ax = pca.plot(model)
    >>> X_norm = pca.norm(X)


    """
    # if sp.issparse(X):
        # if verbose>=1: print('[PCA] Error: A sparse matrix was passed, but dense data is required for method=barnes_hut. Use X.toarray() to convert to a dense numpy array if the array is small enough for it to fit in memory.')
    if sp.issparse(X) and normalize:
        print('[PCA] Can not normalize a sparse matrix.')
        normalize=False
    if isinstance(row_labels, list):
        row_labels=np.array(row_labels)
    if isinstance(col_labels, list):
        col_labels=np.array(col_labels)
    if n_components is None:
        n_components=X.shape[1] - 1
    if col_labels is None or len(col_labels)==0 or len(col_labels)!=X.shape[1]:
        col_labels = np.arange(1,X.shape[1] + 1).astype(str)
    if row_labels is None or len(row_labels)!=X.shape[0]:
        row_labels=np.ones(X.shape[0])
    if (sp.issparse(X) is False) and (n_components>X.shape[1]):
        raise Exception('[PCA] Number of components can not be more then number of features.')

    # normalize data
    if normalize:
        X = preprocessing.scale(X)

    if n_components<1:
        pcp=n_components
        # Run with all components to get all PCs back. This is needed for the step after.
        [model_pca, PC, loadings, percentExplVar] = _explainedvar(X, n_components=None, sparse_data=sparse_data, random_state=random_state)
        # Take nr. of components with minimal expl.var
        n_components= np.min(np.where(percentExplVar>=n_components)[0]) + 1
    else:
        [model_pca, PC, loadings, percentExplVar] = _explainedvar(X, n_components=n_components, sparse_data=sparse_data, random_state=random_state)
        pcp=1

    # Top scoring n_components.
    Iloc = _top_scoring_components(loadings, n_components + 1)

    # Combine components relations with features.
    PCzip = list(zip(['PC-'] * model_pca.components_.shape[0], np.arange(1,model_pca.components_.shape[0] + 1).astype(str)))
    PCnames = list(map(lambda x: ''.join(x), PCzip))
    loadings = pd.DataFrame(loadings, columns=col_labels, index=PCnames)

    # Store
    model = {}
    model['loadings'] = loadings
    model['X'] = PC[:,0:n_components]
    model['explained_var'] = percentExplVar
    model['model'] = model_pca
    model['topn'] = n_components
    model['pcp'] = pcp
    model['col_labels'] = col_labels[Iloc]
    model['y'] = row_labels
    # Return
    return(model)


# %% biplot
def biplot(model, figsize=(10,8)):
    """Create the Biplot based on model.

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
    if model['X'].shape[1]<1:
        raise Exception('[PCA] Requires at least 1 PC to make plot.')
    if len(model['explained_var'])<=1:
        return None, None

    # Get coordinates
    xs = model['X'][:,0]
    if model['X'].shape[1]>1:
        ys = model['X'][:,1]
    else:
        ys = np.zeros(len(xs))

    # Figure
    [fig,ax]=plt.subplots(figsize=figsize, edgecolor='k')
    # Make scatter plot
    uiy=np.unique(model['y'])
    getcolors=np.array(colourmap.generate(len(uiy)))
    for i,y in enumerate(uiy):
        I=(y==model['y'])
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],color=getcolors[i,:], s=25)
        ax.annotate(y, (np.mean(xs[I]), np.mean(ys[I])))

    # Set y
    ax.set_xlabel('PC1 ('+ str(model['model'].explained_variance_ratio_[0] * 100)[0:4] + '% expl.var)')
    ax.set_ylabel('PC2 ('+ str(model['model'].explained_variance_ratio_[1] * 100)[0:4] + '% expl.var)')
    ax.set_title('Biplot\nComponents that cover the [' + str(model['pcp']) + '] explained variance, PC=['+ str(model['topn'])+  ']')
    ax.grid(True)

    # Gather top N loadings
    I = _top_scoring_components(model['loadings'].values, model['topn'])
    xvector = model['loadings'].iloc[0,I]
    yvector = model['loadings'].iloc[1,I]

    # Plot and scale values for arrows and text
    scalex = 1.0 / (model['loadings'].iloc[0,:].max() - model['loadings'].iloc[0,:].min())
    scaley = 1.0 / (model['loadings'].iloc[1,:].max() - model['loadings'].iloc[1,:].min())
    # Plot the arrows
    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        newx=xvector[i] * scalex
        newy=yvector[i] * scaley
        figscaling=np.abs([np.abs(xs).max() / newx, np.abs(ys).max() / newy])
        figscaling=figscaling.min()
        newx=newx * figscaling * 0.5
        newy=newy * figscaling * 0.5
        ax.arrow(0, 0, newx, newy, color='r', width=0.005, head_width=0.05, alpha=0.6)
        ax.text(newx * 1.25, newy * 1.25, model['col_labels'][i], color='black', ha='center', va='center')

    plt.show()
    return(fig, ax)


# %% biplot3d
def biplot3d(model, figsize=(10,8)):
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
    if model['X'].shape[1]<3:
        print('[PCA] Requires 3 PCs to make 3d plot. Auto reverting to: pca.biplot()')
        fig, ax = biplot(model)
        return(fig,ax)
    if len(model['explained_var'])<=1:
        return None, None

    # Get coordinates
    xs = model['X'][:,0]
    ys = model['X'][:,1]
    zs = model['X'][:,2]

    # Figure
    fig = plt.figure(1, figsize=figsize)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # Make scatter plot
    uiy=np.unique(model['y'])
    getcolors=np.array(colourmap.generate(len(uiy)))
    for i,y in enumerate(uiy):
        I=y==model['y']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],zs[I],color=getcolors[i,:], s=25)

    # Set y
    ax.set_xlabel('PC1 ('+ str(model['model'].explained_variance_ratio_[0] * 100)[0:4] + '% expl.var)')
    ax.set_ylabel('PC2 ('+ str(model['model'].explained_variance_ratio_[1] * 100)[0:4] + '% expl.var)')
    ax.set_zlabel('PC3 ('+ str(model['model'].explained_variance_ratio_[2] * 100)[0:4] + '% expl.var)')
    ax.set_title('Components that cover the [' + str(model['pcp']) + '] explained variance, PC=['+ str(model['topn'])+  ']')

    # Gather top N loadings
    I = _top_scoring_components(model['loadings'].values, model['topn'])
    xvector = model['loadings'].iloc[0,I]
    yvector = model['loadings'].iloc[1,I]
    zvector = model['loadings'].iloc[2,I]

    # Plot and scale values for arrows and text
    scalex = 1.0 / (model['loadings'].iloc[0,:].max() - model['loadings'].iloc[0,:].min())
    scaley = 1.0 / (model['loadings'].iloc[1,:].max() - model['loadings'].iloc[1,:].min())
    scalez = 1.0 / (model['loadings'].iloc[2,:].max() - model['loadings'].iloc[2,:].min())
    # Plot the arrows
    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        newx=xvector[i] * scalex
        newy=yvector[i] * scaley
        newz=zvector[i] * scalez
        figscaling=np.abs([np.abs(xs).max() / newx, np.abs(ys).max() / newy])
        figscaling=figscaling.min()
        newx=newx * figscaling * 0.5
        newy=newy * figscaling * 0.5
        newz=newz * figscaling * 0.5
        # ax.arrow(0, 0, newx/20, newy/20, color='r', width=0.0005, head_width=0.005, alpha=0.6)
        ax.text(newx, newy, newz, model['col_labels'][i], color='black', ha='center', va='center')

    plt.show()
    return(fig, ax)


# %% Show explained variance plot
def plot(model, figsize=(10,8)):
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
    # model['model'].explained_variance_ratio_
    explvar = model['explained_var']
    xtick_idx = np.arange(1,len(explvar) + 1)
    [fig,ax]=plt.subplots(figsize=figsize, edgecolor='k')
    plt.plot(xtick_idx, explvar,'o-', color='k', linewidth=1)
    ax.set_xticks(xtick_idx)

    stepsize=2
    xticklabel=xtick_idx.astype(str)
    xticklabel[np.arange(1,len(xticklabel),stepsize)]=''
    ax.set_xticklabels(xticklabel, rotation=90, ha='left', va='top')

    plt.ylabel('Percentage explained variance')
    plt.xlabel('Principle Component')
    plt.ylim([0, 1.05])
    plt.xlim([0, len(explvar) + 1])
    titletxt='Cumulative explained variance\nMinimum components that cover the [' + str(model['pcp']) + '] explained variance, PC=[' + str(model['topn']) + ']'
    plt.title(titletxt)
    plt.grid(True)

    # Plot vertical line To stress the cut-off point
    ax.axhline(y=model['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
    plt.bar(xtick_idx, explvar,color='#3182bd', alpha=0.8)
    plt.show()
    plt.draw()
    return(fig, ax)


# %% Top scoring components
def _top_scoring_components(loadings, topn):
    # Top scoring for 1st component
    I1=np.argsort(np.abs(loadings[0,:]))
    I1=I1[::-1]

    if loadings.shape[0]>=2:
        # Top scoring for 2nd component
        I2=np.argsort(np.abs(loadings[1,:]))
        I2=I2[::-1]
        # Take only top loadings
        I1=I1[0:np.min([topn, len(I1)])]
        I2=I2[0:np.min([topn, len(I2)])]
        I = np.append(I1, I2)
    else:
        I=I1
    # Unique without sort:
    indices = np.unique(I,return_index=True)[1]
    I = [I[index] for index in sorted(indices)]
    return(I)


# %% Top scoring components
def norm(X, n_components=1, pcexclude=[1]):
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
    if  n_components<1:
        raise Exception('n_components must range between [0-1] to select for the nr. of components in explaining variance.')
    if not isinstance(pcexclude,list): pcexclude=[pcexclude]

    # Fit using PCA
    model = fit(X, n_components=X.shape[1])

    coeff = model['loadings'].values
    score = model['X']
    # Compute explained percentage of variance
    q=model['explained_var']
    ndims = np.where(q<=n_components)[0]
    ndims = (np.setdiff1d(ndims + 1,pcexclude)) - 1
    # Transform data
    out = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[1], axis=1) + np.dot(score[:,ndims],coeff[:,ndims].T)
    # Return
    return(out)


# %% Explained variance
def _explainedvar(X, n_components=None, sparse_data=False, random_state=None, n_jobs=-1, verbose=3):
    # Create the model
    if sp.issparse(X):
        if verbose>=3: print('[TruncatedSVD] Fit..')
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif sparse_data:
        if verbose>=3: print('[PCA] Fit sparse dataset..')
        model = SparsePCA(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
    else:
        if verbose>=3: print('[PCA] Fit..')
        model=PCA(n_components=n_components, random_state=random_state)

    # Fit model
    model.fit(X)
    # Do the reduction
    loadings = model.components_ # Ook wel de coeeficienten genoemd: coefs!
    PC = model.transform(X)
    # Compute explained variance, top 95% variance
    percentExplVar = model.explained_variance_ratio_.cumsum()
    # Return
    return(model, PC, loadings, percentExplVar)
