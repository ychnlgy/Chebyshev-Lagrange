import torch, numpy

def rand_indices(n):
    I = torch.arange(n).long()
    numpy.random.shuffle(I.numpy())
    return I
