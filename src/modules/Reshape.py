import torch

class Reshape(torch.nn.Module):
    def __init__(self, *size, contiguous=False, batchsqueeze=False):
        super(Reshape, self).__init__()
        
        if not contiguous:
            self.make_contiguous = self._make_contiguous

        self.batched = batchsqueeze
        self.size = size
    
    def forward(self, X):
        X = self.make_contiguous(X)
        size = [[len(X)], [-1]][self.batched] + list(self.size)
        return X.view(size)
    
    # === PRIVATE ===
    
    def make_contiguous(self, X):
        return X.contiguous()
    
    def _make_contiguous(self, X):
        return X
