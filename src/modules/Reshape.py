import torch

class Reshape(torch.nn.Module):
    def __init__(self, *size, contiguous=False):
        super(Reshape, self).__init__()
        
        if not contiguous:
            self.make_contiguous = self._make_contiguous

        self.size = size
    
    def forward(self, X):
        X = self.make_contiguous(X)
        return X.view(len(X), *self.size)
    
    # === PRIVATE ===
    
    def make_contiguous(self, X):
        return X.contiguous()
    
    def _make_contiguous(self, X):
        return X
