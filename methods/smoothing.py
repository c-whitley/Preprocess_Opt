import numpy as np 
from scipy.signal import savgol_filter
import pywt
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.decomposition import PCA
from .utils import IdentityTransformer
from sklearn.base import TransformerMixin, BaseEstimator


def MakeTransformer(method, **kwargs):

    transformers = {
            'doNothing': IdentityTransformer(),
            'savgol':savgol(),
            'PCA':PCA_smooth()
            } 
    if kwargs: 
        return transformers[method].set_params(**kwargs)
    else: 
        return transformers[method]
    #return transformers[method].set_params(**kwargs)
'''
def getTransformers(): 
    return {
            'doNothing':doNothing,
            'savgol':savgol,
            'wavelet':wavelet,
            'PCA':PCA_smooth
            } 
'''

class savgol(TransformerMixin, BaseEstimator):

    def __init__(self, window = 7, polyorder = 3, **kwargs): 

        self.window = window
        self.polyorder = polyorder

    def fit(self, X, y = None): 

        return self

    def transform(self, X, y = None):
        #print("Performing SG smoothing")
        return pd.DataFrame(savgol_filter(X.values, self.window, self.polyorder), index = X.index, columns = X.columns)


class PCA_smooth(TransformerMixin, BaseEstimator): 

    def __init__(self, n_components = 0.9, **kwargs):

        self.num_components = n_components

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None): 
        #print("Performing PCA smoothing")
        pca = PCA(n_components = self.num_components)

        X_pca = pca.fit_transform(X)

        X_pca = pca.inverse_transform(X_pca)

        X_pca = pd.DataFrame(X_pca, index = X.index, columns = X.columns)

        return X_pca


'''
def wavelet(X, y = None, **kwargs): 

    wavelet = kwargs.get('wavelet','db1')
    X = pd.DataFrame(pywt.dwt(X, wavelet)[0], index = X.index, columns = X.columns[range(len(X.columns))[0::2]])

    return X


def PCA_smooth(X, y = None, **kwargs):

    nc = kwargs.get('nc', 0.95)

    pca = PCA(n_components = nc)

    X_pca = pca.fit_transform(X)

    X_pca = pca.inverse_transform(X_pca)

    X_pca = pd.DataFrame(X_pca, index = X.index, columns = X.columns)

    return X_pca
'''


