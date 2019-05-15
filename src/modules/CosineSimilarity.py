'''

Note:
    The PyTorch CosineSimilarity operation requires that the input
    and target be of the same size. This may be computationally expensive
    if we wish to compare inputs to targets of unit size, since we will
    have to repeat the target to the size of the input in order to use
    the PyTorch implementation of cosine similarity.

    An example where it may be a bottleneck in computation is when we
    wish to apply it to the first convolutional layer.
    
'''

import torch

class CosineSimilarity(torch.nn.Module):

    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, v, w):
        nomin = (v * w).sum(dim=self.dim)
        denom = v.norm(dim=self.dim) * w.norm(dim=self.dim)
        return nomin/(denom+self.eps)

if __name__ == "__main__":

    # Test same functionality

    a = CosineSimilarity()
    b = torch.nn.CosineSimilarity(dim=-1)

    v = torch.rand(3, 5)
    w = torch.rand(3, 5)

    calc1 = a(v, w)
    calc2 = b(v, w)

    def float_equals(v1, v2):
        eps = 1e-6
        return (v1 - v2).norm() < eps

    assert float_equals(calc1, calc2)

    # Test different sizes

    z = torch.rand(1, 5)
    zp = z.repeat(3, 1)

    calc1 = a(v, z) # the new implementation does not need same size
    calc2 = b(v, zp)

    assert float_equals(calc1, calc2)

    # Test differently-sized convolution inputs

    v = torch.rand(2, 3, 5, 5)
    w = torch.rand(1, 3, 1, 1)
    wp = w.repeat(2, 1, 5, 5)

    a = CosineSimilarity(dim=1)
    
    calc1 = a(v, w)
    calc2 = b(v.transpose(1, -1), wp.transpose(1, -1)).transpose(1, -1)

    assert float_equals(calc1, calc2)
