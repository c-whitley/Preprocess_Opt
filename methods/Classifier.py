import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def MakeTransformer(method, **kwargs):

    transformers = {
        'LogisticRegression': LogisticRegression(**kwargs), 
        'Naive Bayes': GaussianNB(**kwargs),
        'Random Forest': RandomForestClassifier(**kwargs)
    }

    return transformers[method]

    