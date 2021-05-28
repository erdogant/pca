from pca import pca
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
import itertools as it
import matplotlib.pyplot as plt
import unittest

class TestPCA(unittest.TestCase):

    def test_plot_combinations(self):

        X = load_iris().data
        labels=load_iris().feature_names
        y=load_iris().target
        
        X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
        
        param_grid = {
        	'n_components':[None, 0.01, 1, 0.95, 2, 100000000000],
        	'row_labels':[None, [], y],
        	'detect_outliers' : [None, 'ht2','spe'],
        	}
        
        allNames = param_grid.keys()
        combinations = it.product(*(param_grid[Name] for Name in allNames))
        combinations=list(combinations)
        
        for combination in combinations:
        	model = pca(n_components=combination[0])
        	model.fit_transform(X)
        	assert model.plot()
        	assert model.biplot(y=y, SPE=True, hotellingt2=True)
        	assert model.biplot3d(y=y, SPE=True, hotellingt2=True)
        	assert model.biplot(y=y, SPE=True, hotellingt2=False)
        	assert model.biplot(y=y, SPE=False, hotellingt2=True)
        	assert model.biplot(y=y, SPE=False, hotellingt2=False)


    def test_correct_ordering_features_in_biplot(self):

        f1=np.random.randint(0,100,250)
        f2=np.random.randint(0,50,250)
        f3=np.random.randint(0,25,250)
        f4=np.random.randint(0,10,250)
        f5=np.random.randint(0,5,250)
        f6=np.random.randint(0,4,250)
        f7=np.random.randint(0,3,250)
        f8=np.random.randint(0,2,250)
        f9=np.random.randint(0,1,250)
        X = np.c_[f1,f2,f3,f4,f5,f6,f7,f8,f9]
        X = pd.DataFrame(data=X, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
        
        model = pca(n_components=9, normalize=False)
        out = model.fit_transform(X)
        assert np.all(out['topfeat'].feature.values == ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'])
        assert model.biplot(n_feat=10, legend=False)
        assert model.biplot3d(n_feat=10, legend=False)
        
        ##### NORMALIZE OUT PC1
        X_norm = model.norm(X, pcexclude=[1])
        X_norm = pd.DataFrame(data=X_norm, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
        out = model.fit_transform(X_norm)
        assert (out['topfeat'].feature.values[-1]=='f1') | (out['topfeat'].feature.values[-2]=='f1')
        assert out['topfeat'].feature.values[0]=='f2'
        
        ##### NORMALIZE OUT PC1 AND PC2
        X_norm = model.norm(X, pcexclude=[1,2])
        X_norm = pd.DataFrame(data=X_norm, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
        out = model.fit_transform(X_norm)
        assert (out['topfeat'].feature.values[-1]=='f2') | (out['topfeat'].feature.values[-1]=='f9') | (out['topfeat'].feature.values[-1]=='f1')
        
        ##### NORMALIZE OUT PC2 AND PC4
        X_norm = model.norm(X, pcexclude=[2])
        X_norm = pd.DataFrame(data=X_norm, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
        out = model.fit_transform(X_norm)
        assert out['topfeat'].feature.values[1]=='f3'
        assert (out['topfeat'].feature.values[-1]=='f2') | (out['topfeat'].feature.values[-1]=='f9')

    def test_for_outliers_and_transparency(self):

        X = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
        outliers = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)
        X = np.vstack((X, outliers))
        
        model = pca(alpha=0.05)
        # Fit transform
        out = model.fit_transform(X)
        assert X[out['outliers']['y_bool'],:].shape[0]==5
        assert out['outliers'].shape[1]==5
        
        ######## TEST FOR HT2 #########
        model = pca(alpha=0.05, detect_outliers=['ht2'])
        # Fit transform
        out = model.fit_transform(X)
        assert X[out['outliers']['y_bool'],:].shape[0]==5
        
        ######## TEST FOR SPE/DMOX #########
        model = pca(alpha=0.05, detect_outliers=['spe'])
        # Fit transform
        out = model.fit_transform(X)
        assert 'y_bool_spe' in out['outliers'].columns
        
        ######## TEST WITHOUT OUTLIERS #########
        model = pca(alpha=0.05, detect_outliers=None)
        # Fit transform
        out = model.fit_transform(X)
        assert out['outliers'].empty
        
        ######## TEST FOR TRANSPARENCY WITH MATPLOTLIB VERSION #########
        assert model.scatter(alpha_transparency=0.1)
        assert model.scatter3d(alpha_transparency=0.1)
        assert model.biplot(alpha_transparency=0.1)
        assert model.biplot3d(alpha_transparency=0.1)
        assert model.scatter(alpha_transparency=None)
        assert model.scatter3d(alpha_transparency=None)
        assert model.biplot(alpha_transparency=None)
        assert model.biplot3d(alpha_transparency=None)
        assert model.scatter(alpha_transparency=0.5)
        assert model.scatter3d(alpha_transparency=0.5)
        assert model.biplot(alpha_transparency=0.5)
        assert model.biplot3d(alpha_transparency=0.5)
        
    def test_for_new_outliers_after_transformation(self):
        # Generate dataset
        np.random.seed(42)
        n_total = 10000
        train_ratio = 0.8
        n_features = 10
        my_array = np.random.randint(low=1, high=10, size=(n_total, n_features))
        features = [f'f{i}' for i in range(1, n_features+1, 1)]
        X = pd.DataFrame(my_array, columns=features)
        X_train = X.sample(frac=train_ratio)
        X_test = X.drop(X_train.index)
        
        # Training
        model = pca(n_components=5, alpha=0.5, n_std=3, normalize=True, random_state=42)
        results = model.fit_transform(X=X_train[features])
        
        # Inference: mapping of data into space.
        PC_test = model.transform(X=X_test[features])
        # Compute new outliers
        scores, _ = model.compute_outliers(PC=PC_test, n_std=3, verbose=3) 
        
        assert model.results.get('outliers_params', None) is not None
        assert scores.shape[0]==X_test.shape[0]

    def test_fit_transform_type(self):
        # Checks that fitting works for expected file types
        X = [[1,2],
             [2,1],
             [3,3]]

        p = pca(n_components=2,normalize=True)
        p.fit_transform(X)
        p.transform(X)
        
