#File containing normalisation methods for pipeline

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, FunctionTransformer, minmax_scale, RobustScaler
from .utils import find_value_num, IdentityTransformer
from sklearn.base import TransformerMixin, BaseEstimator
#methods will return a sklearn FunctionTransformer object which can be incorporated into a pipeline
#X is a pandas dataframe, can have multi-indexing and wavenumber titled columns 
def MakeTransformer(method, **kwargs):

    transformers = {
            'doNothing': IdentityTransformer(),
            'standard': StandardScalerDF(),
            'min_max': min_max(),
            'robust': robust()
            }
    if kwargs: 
        return transformers[method].set_params(**kwargs)
    else:
        
        return transformers[method]
    #return transformers[method].set_params(**kwargs)

class StandardScalerDF(TransformerMixin, BaseEstimator): 

    def __init__(self):

        pass

    def fit(self, X, y = None):
        
        return self

    def transform(self, X, y = None): 
        #print("Performing vector normalisation")
        return pd.DataFrame(StandardScaler().fit_transform(X), index = X.index, columns = X.columns)

class min_max(TransformerMixin, BaseEstimator):

    def __init__(self, **kwargs):

        pass

    def fit(self, X, y = None):

        return self
    
    def transform(self, X, y = None): 
        #print("Performing min-max normalisation")
        return pd.DataFrame(minmax_scale(X, axis = 0), index = X.index, columns = X.columns)

class robust(TransformerMixin, BaseEstimator):

    def __init__(self, **kwargs):

        pass

    def fit(self, X, y = None):

        return self
    
    def transform(self, X, y = None): 
        #print("Performing min-max normalisation")
        return pd.DataFrame(RobustScaler().fit_transform(X), index = X.index, columns = X.columns)
'''
    pd.DataFrame(minmax_scale(X, axis = 1), index = X.index, columns = X.columns)


def vector(X, y = None, **kwargs): 

    X = pd.DataFrame(normalize(X), index = X.index, columns = X.columns)
    
    return X

def min_max(X, y = None, **kwargs): 

    X = pd.DataFrame(minmax_scale(X, axis = 1), index = X.index, columns = X.columns)

    return X

def feature(X, **kwargs):

    feature = kwargs.get('feature', 1655)
    X = X.div(X.iloc[:,find_value_num(feature, X.columns)], axis = 0)

    return X

'''