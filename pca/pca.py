""" This function makes the pca for PCA

	out = pca.biplot(X, <optional>)
    model = pca.fit(X)
          pca.plot(out)
          pca.plot_explainedvar(out)

 INPUT:
   X:              datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction.
                   [2]: (default)

   components=     Float: Take PCs with percentage explained variance>pcp
                   [0.95] (Number of componentens that cover the 95% explained variance)

   topn=           Integer: Show the top n loadings in the figure per principal component. The first PCs are demonstrated thus the unique features in 2x toploadings
                   [25] (default)
                   
   labx=           list of strings of length [x]
                   [] (default)
   
  features=        Numpy Vector of strings: Name of the features that represent the data features and loadings
                   [] (default)
                  
  savemem=         Boolean [False,True]
                   False: No (default)
                   True: Yes (the output of the PCA is directly the embedded space, and not all PCs. This will affect the explained-variance plot)

   height=         Integer:  Height of figure
                   [5]: (default)

   width=          Integer:  Width of figure
                   [5]: (default)

  showfig=         Integer [0,1, 2]
                   [0]: No
                   [1]: Plot explained variance
                   [2]: 2D biplot of 1st and 2nd PC
                   [3]: all of the above
                   
 OUTPUT
	output


 EXAMPLE
   import numpy as np
   from sklearn.datasets import load_iris
   iris = load_iris()
   X=iris.data
   features=np.array(iris.feature_names)
   labx=np.array(iris.target)


   import pca as pca


   # Normalize data by removing PC
   Xnorm = pca.norm(X, pcexclude=[1])
   
   # Make biplot
   out = pca.biplot(Xnorm, components=2, labx=labx, features=features)
   pca.plot_explainedvar(out)
   pca.plot(out)

   model = pca.fit(X)
   out = pca.biplot(X, components=3, labx=labx, features=features)
   pca.plot_explainedvar(out)
   pca.plot(out)
   
   out = pca.biplot(X, components=3, labx=labx, features=features, topn=2)
   pca.plot_explainedvar(out)
   pca.plot(out)

   pca.plot_explainedvar(out)
   pca.plot(out)

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

#%% biplot
def biplot(X, components=2, topn=25, labx=[], features=[], savemem=False):
    '''

    Parameters
    ----------
    X : numpy array
        numpy data array.
    components : Integer, optional
        Number of components to reduce the dimensionality. The default is 2.
    topn : Integer, optional
        Top scoring number. The default is 25.
    labx : String, optional
        Labels of the features. The default is [].
    features : String, optional
        Feature names. The default is [].
    savemem : Bool, optional
        Save memory. The default is False.

    Returns
    -------
    Dictionary.

    '''
    assert topn>0, 'topn requires to be > 0'
    assert  len(labx)==X.shape[0], 'labx should be same size as data'

    Param = dict()
    Param['components']   = components
    Param['topn']         = topn
    Param['savemem']      = savemem
    Param['pcp']          = 0.95 # PCs that cover percentage explained variance
    
    # Set featnames of not exists
    if len(features)==0 or len(features)!=X.shape[1]:
        features = np.arange(1,X.shape[1]+1).astype(str)
    if len(labx)!=X.shape[0]:
        labx=[]
    
    # Fit using PCA
    out = fit(X, components=Param['components'], pcp=Param['pcp'], savemem=Param['savemem'])

    # Set number of components based on PCs with input % explained variance
    if Param['components']<1:
        Param['pcp'] = Param['components']
        Param['components']=out['pc'].shape[1]

    # Top scoring components
    I = top_scoring_components(out['loadings'], Param['topn'])
    # Store
    out['topfeat'] = features[I]
    out['pcp'] = Param['pcp']
    out['topn'] = Param['topn']
    out['labx'] = labx

    # Show explained variance plot
    plot_explainedvar(out)
    # Return
    return(out)

#%% Make PCA fit
def fit(X, components=2, pcp=0.95, savemem=False):
    out=dict()

    # PCA
    if savemem or components<1:
        model=PCA(n_components=components)
    else:
        model=PCA(n_components=X.shape[1])
        
    # Fit
    model.fit(X)
    # Do the reduction
    loadings = model.components_ # Ook wel de coeeficienten genoemd: coefs!
    PC = model.transform(X) # Ook wel SCORE genoemd
    # Compute explained variance, top 95% variance
    percentExplVar = model.explained_variance_ratio_.cumsum()
    pcVar = np.min(np.where(percentExplVar>=pcp)[0])+1  # Plus one because starts with 0
    # Store
    out['loadings'] = loadings
    out['pc'] = PC[:,0:components]
    out['explained_var'] = percentExplVar
    out['pcvar_95'] = pcVar
    out['model'] = model
    # Return
    return(out)

#%% Scatter samples in 2D
def plot(out, height=8, width=10, xlim=[], ylim=[]):
    # Get coordinates
    xs = out['pc'][:,0]
    ys = out['pc'][:,1]

    # Figure
    [fig,ax]=plt.subplots(figsize=(width, height), edgecolor='k')
    # Make scatter plot
    uilabx=np.unique(out['labx'])
    getcolors=discrete_cmap(len(uilabx))
    for i,labx in enumerate(uilabx):
        I=labx==out['labx']
        getcolors[i,:]
        ax.scatter(xs[I],ys[I],color=getcolors[i,:], s=25)
        ax.annotate(labx, (np.mean(xs[I]), np.mean(ys[I])))
    
    # Set labels
    ax.set_xlabel('PC1 ('+ str(out['model'].explained_variance_ratio_[0]*100)[0:4] + '% expl.var)')
    ax.set_ylabel('PC2 ('+ str(out['model'].explained_variance_ratio_[1]*100)[0:4] + '% expl.var)')
    ax.set_title('Biplot of PC1 vs PC2.\nNumber of components to cover the [' + str(out['pcp']) + '] explained variance, PC=['+ str(out['pcvar_95'])+  ']')

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

#%% Show explained variance plot
def plot_explainedvar(out, height=8, width=10, grid=True):
    [fig,ax]=plt.subplots(figsize=(width,height), edgecolor='k')
    plt.plot(np.append(0,out['explained_var']),'o-', color='k')
    plt.ylabel('Percentage explained variance')
    plt.xlabel('Principle Components')
    plt.xticks(np.arange(0,len(out['explained_var'])+1))
    plt.ylim([0,1])
    titletxt='Cumulative explained variance\nMinimum components to cover the [' + str(out['pcp']) + '] explained variance, PC=['+ str(out['pcvar_95'])+  ']'
    plt.title(titletxt)
    plt.grid(grid)

    # Plot vertical line To stress the cut-off point
    # ax.axvline(x=eps[idx], ymin=0, ymax=sillclust[idx], linewidth=2, color='r')
    ax.axhline(y=out['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
    plt.style.use('ggplot')
    plt.bar(np.arange(0,len(out['explained_var'])+1),np.append(0,out['model'].explained_variance_ratio_),color='#3182bd', alpha=0.8)
    plt.show()
    plt.draw()
    
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
    out = fit(X, components=X.shape[1], pcp=1, savemem=savemem)

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