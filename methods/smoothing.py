import numpy as np 
from scipy.signal import savgol_filter
import pywt
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.decomposition import PCA


def MakeTransformer(method, **kwargs):

    transformers = {
            'doNothing':doNothing,
            'savgol':savgol,
            'wavelet':wavelet,
            'PCA':PCA_smooth
            } 
    return FunctionTransformer(transformers[method], kw_args = kwargs)
'''
def getTransformers(): 
    return {
            'doNothing':doNothing,
            'savgol':savgol,
            'wavelet':wavelet,
            'PCA':PCA_smooth
            } 
'''
def doNothing(X, y = None, **kwargs): 
    return X

def savgol(X, y = None, **kwargs):

    window = kwargs.get('window', 7)
    polyorder = kwargs.get('polyorder',3)

    X = pd.DataFrame(savgol_filter(X.values, window, polyorder), index = X.index, columns = X.columns)

    return X

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



