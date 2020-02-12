'''

Data augmentation methods for AD/HC classification.

Assumptions:
    1. Any small set of features are not indicative of the diseased condition.
    2. There exists common features between diseased subjects that are and are
        not relevant to the classification task. Same for healthy subjects.

Possible solutions:
    1. Replace a small, random selection of features with random numbers.
    2. We should swap a large proportion of features between diseased
        individuals. Same for healthy subjects.

'''

import torch

from .. import tensortools

def augment(X0, Y0, perturb_p, split_cond_p, mix_feat_p):
    '''

    Assumes:
        Y is a torch Tensor consisting of 0s and 1s.
    
    '''
    X = X0.clone()
    Y = Y0.clone()
    _perturb(X, perturb_p)
    _mix_between_classes(X, Y, split_cond_p, mix_feat_p)
    #assert (X-X0).norm() > 1e-8
    return X, Y

def _perturb(X, p):
    I = torch.rand_like(X) < p
    X[I] = torch.zeros_like(X)[I].normal_(mean=0, std=1)

def _mix_between_classes(X, Y, sp, mp):
    hc = Y == 0
    ad = Y == 1
    _mix_class(X[hc], sp, mp)
    _mix_class(X[ad], sp, mp)

def _mix_class(X, sp, mp):
    n = int((len(X) * sp)//2)
    I = tensortools.rand_indices(len(X))
    X = X[I]
    X_mix1 = X[:n]
    X_mix2 = X[n:2*n]
    _rand_mix(X_mix1, X_mix2, mp)

def _rand_mix(X_mix1, X_mix2, mp):
    I = torch.rand_like(X_mix1) < mp
    X_mix1[I], X_mix2[I] = X_mix2[I], X_mix1[I]
    
