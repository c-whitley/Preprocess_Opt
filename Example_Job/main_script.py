import numpy as np 
import pandas as pd
import os
from tqdm import tqdm 

import dill

import preprocessing_pipeline as pp 
from methods import *

paramList={
            'binning'   : {'doNothing':{},
                            'MeanBin'  :{'factor': [2,4,8]}
                            },
            'smoothing' : {'doNothing':{},
                            'savgol'   :{'window':[5, 7, 9, 11], 'polyorder':[3, 4, 5]},
                            'PCA'      :{'nc':[5, 10, 20]}
                            },
            'normalise' : { 'vector'   :{},
                            'min_max'  :{},
                            'feature'  :{}
                            },
            'baseline'  : {'doNothing':{},
                            'sg_diff'  :{'window':[5, 7, 9, 11], 'polyorder':[3, 4, 5], 'order':[1, 2]}
                            },
            'FeaExtraction': {'doNothing':{},
                                'PCA'      :{'n_components': [3, 5, 10]},
                                'LDA'      :{'n_components': [1]}
                            },
            'Classifier': {'LogisticRegression': {},
                            'Random Forest': {},
                            'Naive Bayes': {},
                            }
            }

if not os.path.exists('./Input'):
    os.mkdir('./Input')

jobdir = os.getcwd()

# Create loop to make input files for each pipeline
for i, pipe_address in enumerate(pp.BruteForceGenerator(paramList).gen):

    pipe_ob = pp.Pipeline_Opt(pipe_address)

    pipe_ob.make_pipeline()

    with open(os.path.join(jobdir, 'Input', "pipe{}".format(i)), "wb") as f:

        dill.dump(pipe_ob, f, protocol=4)