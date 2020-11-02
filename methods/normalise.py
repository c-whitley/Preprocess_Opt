#File containing normalisation methods for pipeline

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import normalize, minmax_scale, FunctionTransformer
from .utils import find_value_num, IdentityTransformer
from sklearn.base import TransformerMixin, BaseEstimator
#methods will return a sklearn FunctionTransformer object which can be incorporated into a pipeline

#X is a pandas dataframe, can have multi-indexing and wavenumber titled columns 
def MakeTransformer(method, **kwargs):

    transformers = {
            'doNothing': IdentityTransformer(),
            'vector': vector(),
            'min_max': min_max(),
            'feature': feature(),
            }

    return transformers[method].set_params(**kwargs)

class vector(TransformerMixin, BaseEstimator): 

    def __init__(self):

        pass

    def fit(self, X, y = None):
        
        return self

    def transform(self, X, y = None): 

        return pd.DataFrame(normalize(X), index = X.index, columns = X.columns)

class min_max(TransformerMixin, BaseEstimator):

    def __init__(self, **kwargs):

        pass

    def fit(self, X, y = None):

        return self
    
    def transform(self, X, y = None): 

        return pd.DataFrame(minmax_scale(X, axis = 1), index = X.index, columns = X.columns)

class feature(TransformerMixin, BaseEstimator):

    def __init__(self, feature = 1650, **kwargs): 

        self.fea = feature

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None):

        return X.div(X.iloc[:,find_value_num(self.fea, X.columns)], axis = 0)



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