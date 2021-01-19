import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import model_selection, preprocessing
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder
np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4
from collections import Counter, defaultdict
from itertools import compress,chain

class IdentityTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, **kwargs): 

        pass

    def fit(self, X, y = None): 

        return self

    def transform(self, X, y = None): 

        return X 

def stratified_group_k_fold(X, y, groups, n_splits = 5, random_state=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(n_splits)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(random_state).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(n_splits):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(n_splits):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def find_value(value, wavenumbers):
    posit = np.array([])
    val = np.array([])

    for i in range(len(value)):
        
        
        posit = np.append(posit, np.argmin(abs(np.ones(len(wavenumbers))*value[i]-wavenumbers))) 
        val = np.append(val,wavenumbers[int(posit[i])])
    
    (val, posi) = np.unique(val, return_index = True)
    pos = posit[posi]
    
    
    return pos, val

def find_value_num(value, vector): 
    pos = np.argmin(abs(vector.values - value))
    
    return pos

def strings2int(x): 

    uni = np.unique(x)
    x_idx = np.zeros(len(x))
    for i, name in enumerate(uni):
        x_idx[x == name] = i
    
    return x_idx.astype(np.int64), uni

def plot_cv_indices(X, y, group, ax, cv, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
    n_splits = ii + 1
    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('kFold', fontsize=15)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    plt.tight_layout()
    return ax
        

def Split(X, y, split_ob = model_selection.KFold, group = 'patient', random_state = None, **kwargs): 
        """        
        Method which splits the data into training and testing.
        KFoldGroup is custom written splitter that splits into partitions with unique groups i.e. no same patient in two partitions.

        Args:
            split_ob ([type]): [description]
            y (str, optional): [description]. Defaults to 'Class'.
            group (str, optional): [description]. Defaults to 'patient'.
        """
        '''
        if split_ob == 'KFoldGroup':

            y=strings2int(y.values)
            group = strings2int(X.index.get_level_values(group).values)
            ind_gen = stratified_group_k_fold(X.values, y[0], group[0], random_state = random_state,**kwargs)

        else:
        '''     
        ind_gen = split_ob.split(X, y, X.index.get_level_values(group))

        return ind_gen

def visualize_groups(classes, groups, name):
    # Visualize dataset groups

    groups = strings2int(groups)
    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")

class StratifiedGroupKFold:

    def __init__(self, k, random_state=1, balance = False):

        self.random_state = random_state
        self.k = k
        self.balance = balance

    def split(self, X, y, groups):
        
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        groups = encoder.fit_transform(groups)
        #print(y, groups)
        k = self.k #number of splits
        seed = self.random_state #random state
        labels_num = np.max(y) + 1 # number of class labels
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num)) #dictionary has default values of zeros array with n_label elements
        y_distr = Counter() #initialise counter obejct

        for label, g in zip(y, groups):

            y_counts_per_group[g][label] += 1 #adds 1 to index of y counts array corresponding to y
            y_distr[label] += 1 #counts how many in each class

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num)) #initialises dictionary with default values of zeros array with n_label elements
        groups_per_fold = defaultdict(set) #initialises default dictionary with sets as values
        
        def eval_y_counts_per_fold(y_counts, fold): #function takes in y_counts and fold number
            
            #print(i,y_counts)
            
            y_counts_per_fold[fold] += y_counts #adds y_count list from group to fold key in dictionary
            #if fold == 0:
            #print(y_counts_per_fold[fold])
            std_per_label = [] #initialise empty list
            for label in range(labels_num): #iterate through each label (i.e. [1,2] for binary)
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]) #calculates the standard deviation across all the folds for each class
                std_per_label.append(label_std) #append standard deviation to list of labels.
            
            y_counts_per_fold[fold] -= y_counts #remove y counts from array so it resets to zeros
            #print(y_counts_per_fold[3])
            return np.mean(std_per_label) #return the stand
        
        groups_and_y_counts = list(y_counts_per_group.items()) #list of key value tuples, [0] = group, [1] = counts in each class
        random.Random(seed).shuffle(groups_and_y_counts) #shuffles the list of groups

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])): #sorts groups in ascending order of standard deviation of -(class distribution) --> Groups with highest class imbalance first.
            best_fold = None #initialising variables
            min_eval = None 
            for i in range(k): #loops through each fold
                #print("fold", i+1)
                fold_eval = eval_y_counts_per_fold(y_counts, i) #see function
                #print(fold_eval)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

            #print(groups_per_fold)

        all_groups = set(groups)
        #print(all_groups)
        fold_indices = list()
        #print(y_counts_per_fold)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            #train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            #print(np.unique(groups))
            fold_indices.append([i for i, g in enumerate(groups) if g in test_groups])
            #print(max([i for i, g in enumerate(groups) if g in test_groups]))
            #print(np.max(fold_indices))
            #train_indices = group_ind[test_indices != group_ind]
            if self.balance:
                #print(y_counts_per_fold[i])
                num_y = int(np.min(y_counts_per_fold[i]))
                #print(num_y)
                new_fold_indices = []
                for cl in range(labels_num): 
                    #print(type(fold_indices[i]))
                    label_indices = list(compress(fold_indices[i], y[fold_indices[i]] == cl))
                    #print(num_y)
                    
                    label_indices = random.Random(seed).sample(label_indices,num_y)
                    
                    new_fold_indices = new_fold_indices + label_indices
                #print(len(new_fold_indices[1]))    
                fold_indices[i] = new_fold_indices
        

        for i in range(k): 

            test_indices = fold_indices[i]
            #print(max(test_indices))
            train_indices = list(chain.from_iterable([fold_indices[j] for j in range(k) if j is not i]))
            #print(max(train_indices))
            yield train_indices, test_indices
            #yield train_indices, test_indices

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits