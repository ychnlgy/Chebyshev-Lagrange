import torch

class Noise(torch.nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.zeros_like(X).normal_(mean=0, std=self.std)
