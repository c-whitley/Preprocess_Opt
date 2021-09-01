import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def MakeTransformer(method, **kwargs):

    transformers = {
        'LogisticRegression': LogisticRegression(class_weight="balanced", max_iter=500), 
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(class_weight="balanced"), 
        'XGBoost': xgb.XGBClassifier(),
        'LDA': LinearDiscriminantAnalysis()
    }
    if kwargs: 
        return transformers[method].set_params(**kwargs)
    else: 
        return transformers[method]
    #return transformers[method].set_params(**kwargs)
    

    