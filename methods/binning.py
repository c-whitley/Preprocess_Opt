import numpy as np 
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def MakeTransformer(method, **kwargs): 

    transformers = getTransformers()

    return FunctionTransformer(transformers[method], kw_args=kwargs)

def getTransformers():

    return {
            'doNothing': doNothing,
            'MeanBin': MeanBin
            }
            
def doNothing(X, y = None, **kwargs): 

    return X

def MeanBin(X, y = None, **kwargs): 

    factor = kwargs.get('factor', 2)
    ncol_new = X.shape[1]//factor
    overflow = X.shape[1]%factor
    X_new = np.empty((X.shape[0],ncol_new))
    col_new = np.empty(ncol_new)
    
    for i in range(0,ncol_new):
        
        X_new[:,i] = np.mean(X.iloc[:,i*factor:(i*factor+(factor-1))].values, axis = 1)
        col_new[i] = np.round(np.mean(X.columns[i*factor:(i*factor+(factor-1))]))

    if overflow != 0:
        print('overflow = ', overflow) 
        i += 1
        
        X_new = np.append(X_new, np.mean(X.iloc[:, i*factor:].values, axis = 1).reshape(-1,1), axis = 1)
        col_new = np.append(col_new, np.round(np.mean(X.columns[i*factor:])))
    
    X_new = pd.DataFrame(X_new, index = X.index, columns = col_new)
    
    return X_new