import torch

def onehot(I, classes):
    '''

    Input:
        I - torch LongTensor of size (N). Indices for
            onehot encoding.
        classes - int number of columns in the onehot encoding.

    Output:
        onehot - torch FloatTensor of size (N, classes).

    '''
    out = torch.zeros(I.size(0), classes).to(I.device)
    return out.scatter_(1, I.unsqueeze(1), 1)
