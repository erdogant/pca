""" pca is a python package that performs the principal component analysis and to make insightful plots.

    model = pca.fit(X)
	ax    = pca.biplot(model) 
	ax    = pca.biplot3d(model)
	ax    = pca.plot(model)
	Xnorm = pca.norm(X)

 INPUT:
   X:              datamatrix
                   rows    = labels
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction.
                   [2]: (default)

   components=     Float: Take PCs with percentage explained variance>pcp
                   [0.95] (Number of componentens that cover the 95% explained variance)

   y=           list of length [x]. This is only used for coloring
                   [] (default)
   
  labels=          Numpy Vector of strings: Name of the label that represent the data label and loadings
                   [] (default)
                  
   height=         Integer:  Height of figure
                   [5]: (default)

   width=          Integer:  Width of figure
                   [5]: (default)

                   
 OUTPUT
	output


 EXAMPLE
   import numpy as np
   from sklearn.datasets import load_iris
   X = load_iris().data
   labels=iris.feature_names
   y=iris.target


   import pca as pca

   model = pca.fit(X)
   ax = pca.plot(model)
   ax = pca.biplot(model)
   ax = pca.biplot3d(model)
   
   model = pca.fit(X, y=y, labels=labels)
   fig   = pca.biplot(model)
   fig   = pca.biplot3d(model)


   model = pca.fit(X, components=0.95)
   ax = pca.plot(model)
   ax   = pca.biplot(model)

   model = pca.fit(X, components=2)
   ax = pca.plot(model)
   ax   = pca.biplot(model)


   Xnorm = pca.norm(X, pcexclude=[1,2])
   model = pca.fit(Xnorm, y=y, labels=labels)
   ax = pca.biplot(model)
   ax = pca.plot(model)


"""

#--------------------------------------------------------------------------
# Name        : pca.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Nov. 2017
#--------------------------------------------------------------------------

#%% Libraries
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% Explained variance
def _explainedvar(X, components=None):
    # Fit model
    model=PCA(n_components=components)
    model.fit(X)
    # Do the reduction
    loadings = model.components_ # Ook wel de coeeficienten genoemd: coefs!
    PC = model.transform(X)
    # Compute explained variance, top 95% variance
    percentExplVar = model.explained_variance_ratio_.cumsum()
    return(model, PC, loadings, percentExplVar)

#%% Make PCA fit
def fit(X, components=None, y=[], labels=[]):
    '''

    Parameters
    ----------
    X : numpy array
        [NxM] array with columns as features and rows as samples.
    components : [0,..,1] or [1,..number of samples-1] optional
        Number of TOP components to be returned. Values>0 are the number of components. Values<0 are the components that covers at least the percentage of variance
        0.95: (default) Take the number of components that cover at least 95% of variance
        2:    Take the top 2 components
        None: All
    y : [list of integers] optional
        color y that is ued to make the scatterplot
    labels : [list of string] optional
        Numpy Vector of strings: Name of the features that represent the data features and loadings
   

    Returns
    -------
    Dictionary.

    '''

    if isinstance(y, list):
        y=np.array(y)
    if isinstance(labels, list):
        labels=np.array(labels)
    if components is None: 
        components=X.shape[1]
    if len(labels)==0 or len(labels)!=X.shape[1]:
        labels = np.arange(1,X.shape[1]+1).astype(str)
    if len(y)!=X.shape[0]:
        y=np.ones(X.shape[0])

    if components<1:
        pcp=components
        [model, PC, loadings, percentExplVar] = _explainedvar(X)
        components= np.min(np.where(percentExplVar>=components)[0])+1  # Plus one because starts with 0
    else:
        [model, PC, loadings, percentExplVar] = _explainedvar(X, components=components)
        pcp=1

    # Top scoring components
    I = top_scoring_components(loadings, components+1)

    # Store
    out=dict()
    out['loadings'] = loadings
    out['X'] = PC[:,0:components]
    out['explained_var'] = percentExplVar
    out['model'] = model
    out['topn'] = components
    out['pcp'] = pcp
    out['toplabels'] = labels[I]
    out['y'] = y
    # Return
    return(out)

#%% biplot
def biplot(out, height=8, width=10, xlim=[], ylim=[]):
    assert out['X'].shape[1]>0, print('[PCA] Requires at least 1 PC to make plot.')

    # Get coordinates
    xs = out['X'][:,0]
    ys = out['X'][:,1]

    # Figure
    [fig,ax]=plt.subplots(figsize=(width, height), edgecolor='k')
    # Make scatter plot
    uiy=np.unique(out['y'])
    getcolors=_discrete_cmap(len(uiy))
    for i,y in enumerate(uiy):
        I=y==out['y']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],color=getcolors[i,:], s=25)
        ax.annotate(y, (np.mean(xs[I]), np.mean(ys[I])))
    
    # Set y
    ax.set_xlabel('PC1 ('+ str(out['model'].explained_variance_ratio_[0]*100)[0:4] + '% expl.var)')
    ax.set_ylabel('PC2 ('+ str(out['model'].explained_variance_ratio_[1]*100)[0:4] + '% expl.var)')
    ax.set_title('Biplot\nComponents that cover the [' + str(out['pcp']) + '] explained variance, PC=['+ str(out['topn'])+  ']')
    ax.grid(True)

    #% Gather top N loadings
    I = top_scoring_components(out['loadings'], out['topn'])
    xvector = out['loadings'][0,I]
    yvector = out['loadings'][1,I]
    
    # Plot and scale values for arrows and text
    scalex = 1.0/(out['loadings'][0,:].max() - out['loadings'][0,:].min())
    scaley = 1.0/(out['loadings'][1,:].max() - out['loadings'][1,:].min())
    # Plot the arrows
    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        newx=xvector[i]*scalex
        newy=yvector[i]*scaley
        figscaling=np.abs([np.abs(xs).max()/newx, np.abs(ys).max()/newy])
        figscaling=figscaling.min()
        newx=newx*figscaling*0.5
        newy=newy*figscaling*0.5
        ax.arrow(0, 0, newx, newy, color='r', width=0.005, head_width=0.05, alpha=0.6)
        ax.text(newx*1.25, newy*1.25, out['toplabels'][i], color='black', ha='center', va='center')

    plt.show()
    return(ax)

#%% biplot3d
def biplot3d(out, height=8, width=10, xlim=[], ylim=[]):
    assert out['X'].shape[1]>2, print('[PCA] Requires 3 PCs to make 3d plot. Try pca.biplot()')

    # Get coordinates
    xs = out['X'][:,0]
    ys = out['X'][:,1]
    zs = out['X'][:,2]

    # Figure
    fig = plt.figure(1, figsize=(width, height))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # Make scatter plot
    uiy=np.unique(out['y'])
    getcolors=_discrete_cmap(len(uiy))
    for i,y in enumerate(uiy):
        I=y==out['y']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],zs[I],color=getcolors[i,:], s=25)

    # Set y
    ax.set_xlabel('PC1 ('+ str(out['model'].explained_variance_ratio_[0]*100)[0:4] + '% expl.var)')
    ax.set_ylabel('PC2 ('+ str(out['model'].explained_variance_ratio_[1]*100)[0:4] + '% expl.var)')
    ax.set_zlabel('PC3 ('+ str(out['model'].explained_variance_ratio_[2]*100)[0:4] + '% expl.var)')
    ax.set_title('Components that cover the [' + str(out['pcp']) + '] explained variance, PC=['+ str(out['topn'])+  ']')

    #% Gather top N loadings
    I = top_scoring_components(out['loadings'], out['topn'])
    xvector = out['loadings'][0,I]
    yvector = out['loadings'][1,I]
    zvector = out['loadings'][2,I]
    
    # Plot and scale values for arrows and text
    scalex = 1.0/(out['loadings'][0,:].max() - out['loadings'][0,:].min())
    scaley = 1.0/(out['loadings'][1,:].max() - out['loadings'][1,:].min())
    scalez = 1.0/(out['loadings'][2,:].max() - out['loadings'][2,:].min())
    # Plot the arrows
    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        newx=xvector[i]*scalex
        newy=yvector[i]*scaley
        newz=zvector[i]*scalez
        figscaling=np.abs([np.abs(xs).max()/newx, np.abs(ys).max()/newy])
        figscaling=figscaling.min()
        newx=newx*figscaling*0.5
        newy=newy*figscaling*0.5
        newz=newz*figscaling*0.5
        # ax.arrow(0, 0, newx/20, newy/20, color='r', width=0.0005, head_width=0.005, alpha=0.6)
        ax.text(newx, newy, newz, out['toplabels'][i], color='black', ha='center', va='center')

    plt.show()
    return(ax)

#%% Show explained variance plot
def plot(out, height=8, width=10):
    [fig,ax]=plt.subplots(figsize=(width,height), edgecolor='k')
    plt.plot(np.append(0,out['explained_var']),'o-', color='k', linewidth=1)
    plt.ylabel('Percentage explained variance')
    plt.xlabel('Principle Component')
    plt.xticks(np.arange(0,len(out['explained_var'])+1))
    plt.ylim([0,1])
    titletxt='Cumulative explained variance\nMinimum components that cover the [' + str(out['pcp']) + '] explained variance, PC=['+ str(out['topn'])+  ']'
    plt.title(titletxt)
    plt.grid(True)

    # Plot vertical line To stress the cut-off point
    # ax.axvline(x=eps[idx], ymin=0, ymax=sillclust[idx], linewidth=2, color='r')
    ax.axhline(y=out['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
    plt.bar(np.arange(0,len(out['explained_var'])+1),np.append(0,out['model'].explained_variance_ratio_),color='#3182bd', alpha=0.8)
    plt.show()
    plt.draw()
    return(ax)
    
#%% Top scoring components
def top_scoring_components(loadings, topn):
    # Top scoring for 1st component
    I1=np.argsort(np.abs(loadings[0,:]))
    I1=I1[::-1]
    # Top scoring for 2nd component
    I2=np.argsort(np.abs(loadings[1,:]))
    I2=I2[::-1]
    # Take only top loadings
    I1=I1[0:np.min([topn,len(I1)])]
    I2=I2[0:np.min([topn,len(I2)])]
    I = np.append(I1,I2)
    # Unique without sort:
    indices=np.unique(I,return_index=True)[1]
    I = [I[index] for index in sorted(indices)]
    return(I)

#%% Top scoring components
def norm(X, pcp=1, pcexclude=[1], savemem=False):
    '''
    Normalize your data using the principal components.
    As an example, suppose there is (technical) variation in the fist
    component and you want that out. This function transforms the data using
    the components that you want, e.g., starting from the 2nd pc, up to the
    pc that contains at least 95% of the explained variance
    '''

    assert pcp<=1, 'pcp must range between [0-1]'
    if not isinstance(pcexclude,list): pcexclude=[pcexclude]

    # Fit using PCA
    out = fit(X, components=X.shape[1])

    coeff = out['loadings']
    score = out['X']
    # Compute explained percentage of variance
    q=out['explained_var']
    ndims = np.where(q<=pcp)[0]
    ndims = (np.setdiff1d(ndims+1,pcexclude))-1
    # Transform data
    out = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[1], axis=1) + np.dot(score[:,ndims],coeff[:,ndims].T)
    # Return
    return(out)

#%%
def _discrete_cmap(N, cmap='Set1'):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(cmap)
    color_list = base(np.linspace(0, 1, N))
    return color_list
