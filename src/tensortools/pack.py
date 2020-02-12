import numpy

def pack(arrays):

    '''

    Input:
        arrays - list of numpy array of shape (N, D) or (N).

    Output:
        A - numpy array of shape (N, D'), where D' is the sum
            of all features across all matrices, where the
            feature count of a vector is 1.

    '''
    
    arrays = _expand_vectors(arrays)
    return numpy.concatenate(arrays, axis=1)

def _expand_vectors(arrays):
    return [
        numpy.expand_dims(arr, axis=1) 
        if len(arr.shape) == 1 else arr
        for arr in arrays 
    ]
