import torch.utils

def create_loader(tensors, **kwargs):
    dataset = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader
