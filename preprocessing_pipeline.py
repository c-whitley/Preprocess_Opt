import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from pprint import pprint

import itertools
import collections
import sys

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import model_selection 
from sklearn import metrics 

from .methods import binning, normalise, smoothing, baseline, FeaExtraction, Classifier, utils

class Pipeline_Opt:
    """[summary]
    """    

    def __init__(self, address):
        """[summary]
        """        

        self.mods = {
            'binning':binning, 
            'smoothing': smoothing,
            'normalise': normalise,
            'baseline': baseline,
            'FeaExtraction': FeaExtraction,
            'Classifier': Classifier
        }

        self.make_pipeline()


    def make_pipeline(self):
        """[summary]

        Args:
            address ([type]): [description]
        """
        #input an dictionary indicating which method and parameter to use for each step         
        
        pipeline_list = []
        transformer = []
        
        for k, v in self.address.items():
            transformer = []
            transformer.append(k)
            
            transformer.append(self.mods[k].MakeTransformer(v[1], **v[2]))
            
            pipeline_list.append(tuple(transformer))
            
            self.pipeline = Pipeline(pipeline_list)


    def transform_data(self, X, y):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
        """        
         
        self.X_t = self.pipeline.fit_transform(X, y)

        if not(isinstance(self.X_t, pd.DataFrame)):

            if isinstance(self.pipeline.get_params().get('FeaExtraction'), PCA):
                lab = "PC"
            elif isinstance(self.pipeline.get_params().get('FeaExtraction'), LDA):
                lab = "LD"

            cols = ["{} {}".format(lab, i) for i in range(1, self.pipeline.get_params().get('FeaExtraction__n_components')+1)]

            self.X_t = pd.DataFrame(self.X_t, columns = cols, index = X.index)


    def trial(self, X, y, pos, **kwargs):
        """
        Method that appends a classifer to the end of the pipeline and runs a trial

        Args:
            X ([type]): [description]
            y ([type]): [description]
            pos ([type]): [description]

        Returns:
            [type]: [description]
        """        
        self.X = X

        #split object of module 'sklearn.model_selection'
        split_ob = kwargs.get('split', model_selection.KFold(n_splits=5, random_state=3, shuffle=True))

        #sklearn classifier object
        #cl = kwargs.get('cl', RF(random_state=True))

        #transform X with pipeline
        #self.X_t = self.pipeline.fit_transform(X, y)
        #self.transform_data(self.X, y)


        self.Split(split_ob)

        scores = model_selection.cross_validate(self.pipeline, self.X, y, cv = self.ind_gen, scoring = {'auc':'roc_auc', 'sensitivity':metrics.make_scorer(metrics.recall_score, pos_label = pos)})#, pos_label=pos), 'auc':metrics.make_scorer(metrics.roc_auc_score), 'precision':metrics.make_scorer(metrics.precision_score, pos_label='T')})
           
        #scores = model_selection.cross_validate(cl, self.X_t, y, cv = self.ind_gen, scoring = {'auc':'roc_auc', 'sensitivity':metrics.make_scorer(metrics.recall_score, pos_label = pos)})#, pos_label=pos), 'auc':metrics.make_scorer(metrics.roc_auc_score), 'precision':metrics.make_scorer(metrics.precision_score, pos_label='T')})

        return scores


    def Split(self, split_ob, y = 'Class', group = 'patient', **kwargs): 
        """        
        Method which splits the data into training and testing.
        KFoldGroup is custom written splitter that splits into partitions with unique groups i.e. no same patient in two partitions.

        Args:
            split_ob ([type]): [description]
            y (str, optional): [description]. Defaults to 'Class'.
            group (str, optional): [description]. Defaults to 'patient'.
        """
        
        if split_ob == 'KFoldGroup':

            y = utils.strings2int(self.X.index.get_level_values(y).values)
            group = utils.strings2int(self.X.index.get_level_values(group).values)
            self.ind_gen = utils.stratified_group_k_fold(self.X.values, y[0], group[0], **kwargs)

        else: 
            self.ind_gen = split_ob.split(self.X, self.X.index.get_level_values(y), self.X.index.get_level_values(group))

#----------------------------------------------------------------------------------------------------------------------------------

class BruteForceGenerator:
    """
    Class for the bruce force generation of pipelines.

    Yields:
        [type]: [description]
    """    """
    
    """    

    def __init__(self, paramList, **kwargs):
        """[summary]

        Args:
            paramList ([type]): [description]
        """
        self.paramList = paramList
        self.order = kwargs.get('order',['binning','smoothing','normalise','baseline','FeaExtraction', 'Classifier'])
        self.gen = self.get_addresses()
        

    def get_options(self, input_dict):
        """[summary]

        Args:
            input_dict ([type]): [description]

        Yields:
            [type]: [description]
        """

        for step, functions in input_dict.items():
            for function, kw_dict in functions.items():
                if len(kw_dict)==0:
                    
                    # Yield the step with no keyword arguments
                    yield [step, function, {}]

                # Check if we're down to the level of the parameter values
                elif all([isinstance(vals, list) for vals in kw_dict.values()]):

                    # Get all of the possible combinations for the function's arguments
                    combos = np.array(np.meshgrid(*kw_dict.values())).reshape(len(kw_dict.values()),-1).T

                    for combination in combos:

                        kwargs = dict(zip(kw_dict.keys() ,combination))

                        yield [step, function, kwargs]


    def get_addresses(self):

        """[summary]

        Yields:
            [type]: [description]
        """        

        input_dict = collections.OrderedDict({key: self.paramList[key] for key in self.order})

        # Get list of possible permutations
        permutations = list(self.get_options(input_dict))

        # Count how many different permutations are in each step
        func_counts = [f[0] for f in permutations]
        ls=[p[0] for p in permutations]
        lookup = set()
        ls = [x for x in ls if x not in lookup and lookup.add(x) is None]


        n_params = {step: func_counts.count(step) for step in ls}

        # Generate all possible addresses for each permutation
        addresses = np.array(list(itertools.product(*[np.arange(int(c)) for c in n_params.values()]))).squeeze()

        ls=[p[0] for p in permutations]
        lookup = set()
        ls = [x for x in ls if x not in lookup and lookup.add(x) is None]

        sorted_perms = collections.OrderedDict(((step, [perm for perm in permutations if perm[0]==step]) for step in ls))


        for i in range(addresses.shape[0]):

            # Get an address for the permutation from the complete list
            address = addresses[i,:]

            # Return permutations one at a time
            yield {step: sorted_perms[step][perm] for step, perm in zip(ls, address)}