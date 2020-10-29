#File containing normalisation methods for pipeline

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import normalize, minmax_scale, FunctionTransformer
from .utils import find_value_num
#methods will return a sklearn FunctionTransformer object which can be incorporated into a pipeline

#X is a pandas dataframe, can have multi-indexing and wavenumber titled columns 
def MakeTransformer(method, **kwargs):

    transformers = {
            'doNothing': doNothing,
            'vector': vector,
            'min_max': min_max,
            'feature': feature,
            }

    return FunctionTransformer(transformers[method], kw_args=kwargs)
'''    
def getTransformers():  

    return {
            'doNothing': doNothing,
            'vector': vector,
            'min_max': min_max,
            'feature': feature,
            }
'''
def doNothing(X, y=None, **kwargs):
    return X

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

