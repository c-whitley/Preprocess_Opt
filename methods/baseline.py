import numpy as np 
import pandas as pd 
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import ray
from scipy.spatial import ConvexHull
from .utils import IdentityTransformer


def MakeTransformer(method, **kwargs):
    
    
    transformers = {
                    'doNothing': IdentityTransformer(),
                    'sg_diff': sg_diff(),
                    'rubberband': Rubber_Band()
                    }


    return transformers[method].set_params(**kwargs)
    

#Savitzy Golay differentiation 
class sg_diff(TransformerMixin, BaseEstimator):

    def __init__(self, window = 7, polyorder = 3, order = 1): 

        self.window = window
        self.polyorder = polyorder
        self.order = order

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None):

        return pd.DataFrame(savgol_filter(X.values, self.window, self.polyorder, self.order), index=X.index, columns = X.columns)

class Rubber_Band(TransformerMixin, BaseEstimator):

    def __init__(self, num_jobs = 4): 

        self.n_jobs = num_jobs

    def fit(self, X, y = None):
        ray.shutdown() 
        ray.init(num_cpus=self.n_jobs)
        self.baseline = np.array(ray.get([rubberband_fitter.remote(i) 
        for i in np.apply_along_axis(lambda row: row, axis = 0, arr=X)]))
        return self

    def transform(self, X, y = None): 

        return X - self.baseline



    


'''
class BaselineCorrection(TransformerMixin, BaseEstimator): 

    def __init__(self, method, **kwargs): 
        self.kw_args = kwargs
        self.method = method

    def fit(self, X, y = None): 
        return self

    def transform(self, X, y = None):

        if self.method == "doNothing": 

            return X

        if self.method == "rubberband":

            return rubberband(X, 4)

        if self.method == "sg_diff": 
            
            return sg_diff(X, self.kw_args)
'''
@ray.remote
def rubberband_fitter(spectrum):

    wn = np.arange(len(spectrum))
    points = np.column_stack([wn, spectrum])

    verts = ConvexHull(points).vertices

    # Rotate convex hull vertices until they start from the lowest one
    verts = np.roll(verts, -verts.argmin())
    # Leave only the ascending part
    verts = verts[:verts.argmax()]

    baseline = np.interp(wn, wn[verts], spectrum[verts])

    return baseline
'''
def rubberband(X, n_jobs):

    ray.init(num_cpus=n_jobs)
    baseline = np.array(ray.get([rubberband_fitter.remote(i) 
        for i in np.apply_along_axis(lambda row: row, axis = 0, arr=X)]))
    
    return X - baseline

def doNothing(X, y = None, **kwargs): 
    return X

def sg_diff(X, kwargs): 

    window=kwargs.get('window', 7)
    polyorder=kwargs.get('polyorder', 3)
    order=kwargs.get('order', 1)

    X = pd.DataFrame(savgol_filter(X.values, window, polyorder, order), index=X.index, columns = X.columns)
    
    return X
'''