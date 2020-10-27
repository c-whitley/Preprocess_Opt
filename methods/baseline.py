import numpy as np 
import pandas as pd 
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer

def MakeTransformer(method, **kwargs): 
    
    transformers = getTransformers()

    return FunctionTransformer(transformers[method], kw_args=kwargs)


#Savitzy Golay differentiation 
def getTransformers(): 

    return {
        'doNothing':doNothing,
        'sg_diff':sg_diff
        }

def doNothing(X, y = None, **kwargs): 
    return X

def sg_diff(X, y = None, **kwargs): 

    window=kwargs.get('window', 7)
    polyorder=kwargs.get('polyorder', 3)
    order=kwargs.get('order', 1)

    X = pd.DataFrame(savgol_filter(X.values, window, polyorder, order), index=X.index, columns = X.columns)
    
    return X
