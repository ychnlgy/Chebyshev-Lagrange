import src

import collections, numpy, random

import sklearn.model_selection

def k_fold(U, X, Y, k, seed=None):
    
    '''
    
    Input:
        U - numpy array of (N) shape, the unique subject ids.
        X - numpy array of (N, D) shape, the features.
        Y - numpy array of (N) shape, the labels.
        k - int number of folds.
    
    Output:
        iter of (
            (X_train, Y_train),
            (X_valid, Y_valid)
        ) - the training and test set per fold.
    
    '''
    if seed is not None:
        random.seed(seed)
        
    X = src.tensortools.pack([U, X, Y])
    
    bins = _collect_by_uid(X, uid_column=0)
    kfolds = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=seed)
    for data_i, test_i in kfolds.split(bins):

        data = _concat(bins[data_i])
        test = _concat(bins[test_i])
        
        data_X = data[:,1:-1]
        test_X = test[:,1:-1]
        
        yield (
            (data_X, data[:,-1].astype(numpy.int64)),
            (test_X, test[:,-1].astype(numpy.int64))
        )

def _concat(listofarrays):
    return numpy.array([row for group in listofarrays for row in group])

def _collect_by_uid(X, uid_column):
    
    '''
    
    Input:
        X - numpy array of (N, D) shape
        uid_column - int in [0, D), index of unique ids.
    
    Output:
        bins - list of rows per unique id.
    
    '''
    
    bins = collections.defaultdict(list)
    for row in X:        
        bins[int(row[uid_column])].append(row)
    out = list(bins.values())
    random.shuffle(out)
    return numpy.array(out, dtype=numpy.object)
