import torch

class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.rand_like(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.rand_like(grad_output)
        return beta * grad_output, (1 - beta) * grad_output, None

class ShakeBlock(torch.nn.Module):

    def __init__(self, block1, block2, shortcut=torch.nn.Sequential()):
        super(ShakeBlock, self).__init__()
        self.bk1 = block1
        self.bk2 = block2
        self.sc = shortcut
        
    def forward(self, X):
        h1 = self.bk1(X)
        h2 = self.bk2(X)
        h = ShakeShake.apply(h1, h2, self.training)
        return self.sc(X) + h
