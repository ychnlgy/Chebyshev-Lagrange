import torch

def paramcount(m):
    return sum(torch.numel(p) for p in m.parameters() if p.requires_grad)
