from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd
from .utils import find_value, IdentityTransformer

def MakeTransformer(method, **kwargs): 

    transformers = {
                    'doNothing': IdentityTransformer(),
                    'Truncate': Truncate()
                    }
    if kwargs: 
        return transformers[method].set_params(**kwargs)
    else: 
        return transformers[method]

class Truncate( BaseEstimator , TransformerMixin ): 

    def __init__(self, remove = [(1340,1490), (2300,2400), (2700,3000)], ends = "fingerprint"): 


        self.remove = remove
        self.ends = ends 

    def fit(self, X, y=None, **kwargs): 

        return self

    def transform(self, X, y=None, **kwargs):
        
        if self.remove:

            for region in self.remove: 

                pos, val = find_value(region, X.columns.values)
                X = X.drop(X.columns[int(pos[0]):int(pos[1])], axis = 1)

        if self.ends == "fingerprint":
                   
            pos, val = find_value([1000,1800], X.columns.values)
            X = X.iloc[:, int(pos[0]):int(pos[1])]

        return X

    
            
