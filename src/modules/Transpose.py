import torch

class Transpose(torch.nn.Module):

    def __init__(self, dim1, dim2):
        super().__init__()
        self.d1 = dim1
        self.d2 = dim2

    def forward(self, X):
        return X.transpose(self.d1, self.d2)
