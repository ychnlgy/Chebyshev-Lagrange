import torch

class PrototypeClassifier(torch.nn.Module):

    def __init__(self, features, classes):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.prototype = torch.nn.Linear(features, classes, bias=False)
        
    def forward(self, X):
        return self.softmax(self.prototype(X))
