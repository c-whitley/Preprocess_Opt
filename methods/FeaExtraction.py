import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import FunctionTransformer


def MakeTransformer(method, **kwargs):

    transformers = {
        'doNothing': FunctionTransformer(doNothing),
        'PCA': PCA(**kwargs),
        'LDA': LinearDiscriminantAnalysis(**kwargs)
    }
    
    return transformers[method]
    #return FunctionTransformer(transformers[method], kw_args=kwargs)

'''
def getTransformers(): 

    return 
'''

def doNothing(X,y=None,**kwargs):

    return X

'''   
def pca(X, y= None, **kwargs):

    nc = kwargs.get('nc',3)
    pca_model = PCA(n_components = nc)
    X_transform = pca_model.fit_transform(X)
    cols = ["PC{:d}".format(i+1) for i in range(X_transform.shape[1])]

    return pd.DataFrame(X_transform, index = X.index, columns=cols)

def lda(X, y, **kwargs): 

    nc = kwargs.get('nc', 3)
    lda_model = LinearDiscriminantAnalysis(n_components = nc)
    X_transform = lda_model.fit_transform(X.values, y)
    cols = ["LD{:d}".format(i+1) for i in range(X_transform.shape[1])]
    return pd.DataFrame(X_transform, index=X.index, columns=cols)
'''
    
