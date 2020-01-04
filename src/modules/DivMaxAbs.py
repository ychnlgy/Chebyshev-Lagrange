import torch

class DivMaxAbs(torch.nn.Module):

    def __init__(self, momentum, lr, initial_value=1, eps=1e-16):
        super().__init__()
        self.p = momentum
        self.v = initial_value
        self.e = eps
        self.lr = lr

    def forward(self, X):
        T = X.view(X.size(0), X.size(1), -1)
        out = T/(self.v + self.e)
        
        if self.training:
            A = X.abs()
            M = X.max(dim=0)[0].max(dim=-1)[0].clone().detach()
            v = self.v * self.p + M * (1-self.v)
            self.v += self.lr * (v - self.v)
        
        return out.view(X.size())
