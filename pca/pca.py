""" This function makes the pca for PCA

    model = pca.fit(X)
	out   = pca.biplot(X, <optional>)
            pca.scatterplot(out)
            pca.plot_explainedvar(out)

 INPUT:
   X:              datamatrix
                   rows    = feat
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction.
                   [2]: (default)

   components=     Float: Take PCs with percentage explained variance>pcp
                   [0.95] (Number of componentens that cover the 95% explained variance)

   labels=           list of strings of length [x]
                   [] (default)
   
  feat=        Numpy Vector of strings: Name of the feat that represent the data feat and loadings
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
   iris = load_iris()
   X=iris.data
   feat=iris.feature_names
   labels=iris.target


   import pca as pca

   model = pca.fit(X)
   ax = pca.plot_explainedvar(model)
   a  = pca.biplot(model)
   a  = pca.biplot3d(model)
   
   fig   = pca.biplot(model)
   fig   = pca.biplot3d(model)

   model = pca.fit(X, labels=labels, feat=feat)
   fig   = pca.biplot(model)
   fig   = pca.biplot3d(model)


   model = pca.biplot(X, components=None, labels=labels, feat=feat)
   pca.scatterplot(model)
   pca.scatterplot3d(model)

   model = pca.fit(X)
   pca.scatterplot(model)

   model = pca.biplot(X, components=2)
   pca.scatterplot(model)

   model = pca.biplot(X, components=0.95)
   pca.scatterplot(model)

   model = pca.biplot(X, components=0.95, labels=labels, feat=feat)
   pca.scatterplot(model)


   # Normalize data by removing PC
   Xnorm = pca.norm(X, pcexclude=[1,2,3,4])
   model = pca.biplot(Xnorm, components=None, labels=labels, feat=feat)
   pca.scatterplot(model)

   Xnorm = pca.norm(X, pcexclude=[1,2,3])
   model = pca.biplot(Xnorm, components=None, labels=labels, feat=feat)
   pca.scatterplot(model)

   Xnorm = pca.norm(X, pcexclude=[1,2])
   model = pca.biplot(Xnorm, components=None, labels=labels, feat=feat)
   pca.scatterplot(model)

   Xnorm = pca.norm(X, pcexclude=[1])
   model = pca.biplot(Xnorm, components=None, labels=labels, feat=feat)
   pca.scatterplot(model)

   
   pca.plot_explainedvar(out)

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
def explainedvar(X, components=None):
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
def fit(X, components=None, labels=[], feat=[]):
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
    labels : [list of integers] optional
        color label that is ued to make the scatterplot
    feat : [list of string] optional
        Numpy Vector of strings: Name of the features that represent the data features and loadings
   

    Returns
    -------
    Dictionary.

    '''

    if isinstance(labels, list):
        labels=np.array(labels)
    if isinstance(feat, list):
        feat=np.array(feat)
    if components is None: 
        components=X.shape[1]
    if len(feat)==0 or len(feat)!=X.shape[1]:
        feat = np.arange(1,X.shape[1]+1).astype(str)
    if len(labels)!=X.shape[0]:
        labels=np.ones(X.shape[0])

    if components<1:
        pcp=components
        [model, PC, loadings, percentExplVar] = explainedvar(X)
        components= np.min(np.where(percentExplVar>=components)[0])+1  # Plus one because starts with 0
    else:
        [model, PC, loadings, percentExplVar] = explainedvar(X, components=components)
        pcp=1

    # Top scoring components
    I = top_scoring_components(loadings, components+1)

    # Store
    out=dict()
    out['loadings'] = loadings
    out['pc'] = PC[:,0:components]
    out['explained_var'] = percentExplVar
    out['model'] = model
    out['topn'] = components
    out['pcp'] = pcp
    out['topfeat'] = feat[I]
    out['labels'] = labels
    # Return
    return(out)

#%% biplot
def biplot(out, height=8, width=10, xlim=[], ylim=[]):
    assert out['pc'].shape[1]>0, print('[PCA] Requires at least 1 PC to make plot.')

    # Get coordinates
    xs = out['pc'][:,0]
    ys = out['pc'][:,1]

    # Figure
    [fig,ax]=plt.subplots(figsize=(width, height), edgecolor='k')
    # Make scatter plot
    uilabel=np.unique(out['labels'])
    getcolors=discrete_cmap(len(uilabel))
    for i,label in enumerate(uilabel):
        I=label==out['labels']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],color=getcolors[i,:], s=25)
        ax.annotate(label, (np.mean(xs[I]), np.mean(ys[I])))
    
    # Set labels
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
        ax.text(newx*1.25, newy*1.25, out['topfeat'][i], color='black', ha='center', va='center')

    plt.show()
    plt.draw()
    return(ax)

#%% biplot3d
def biplot3d(out, height=8, width=10, xlim=[], ylim=[]):
    assert out['pc'].shape[1]>2, print('[PCA] Requires 3 PCs to make 3d plot. Try pca.biplot()')

    # Get coordinates
    xs = out['pc'][:,0]
    ys = out['pc'][:,1]
    zs = out['pc'][:,2]

    # Figure
    fig = plt.figure(1, figsize=(width, height))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # Make scatter plot
    uilabel=np.unique(out['labels'])
    getcolors=discrete_cmap(len(uilabel))
    for i,label in enumerate(uilabel):
        I=label==out['labels']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],zs[I],color=getcolors[i,:], s=25)

    # Set labels
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
        ax.arrow(0, 0, newx/20, newy/20, color='r', width=0.0005, head_width=0.005, alpha=0.6)
        ax.text(newx, newy, newz, out['topfeat'][i], color='black', ha='center', va='center')

    plt.show()
    plt.draw()
    return(ax)

#%% Show explained variance plot
def plot_explainedvar(out, height=8, width=10):
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
    score = out['pc']
    # Compute explained percentage of variance
    q=out['explained_var']
    ndims = np.where(q<=pcp)[0]
    ndims = (np.setdiff1d(ndims+1,pcexclude))-1
    # Transform data
    out = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[1], axis=1) + np.dot(score[:,ndims],coeff[:,ndims].T)
    # Return
    return(out)

#%%
def discrete_cmap(N, cmap='Set1'):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(cmap)
    color_list = base(np.linspace(0, 1, N))
    return color_list
