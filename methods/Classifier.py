import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def MakeTransformer(method, **kwargs):

    transformers = {
        'LogisticRegression': LogisticRegression(), 
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier()
    }

    return transformers[method].set_params(**kwargs)

    