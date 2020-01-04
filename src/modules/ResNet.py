import torch

class ResNet(torch.nn.Module):

    def __init__(self, *blocks):
        super(ResNet, self).__init__()
        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        return X
        
