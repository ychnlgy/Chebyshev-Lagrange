import torch

class PredictorGroup(torch.nn.Module):

    def __init__(self, nets, combiner):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.combiner = combiner

    def forward(self, X):
        N, D = X.size()
        P = len(self.nets)
        assert (D % P) == 0
        X = X.view(N, P, -1)
        Y = torch.stack(list(self._iter_forward(X)), dim=-1)
        return self.combiner(Y).squeeze(-1)

    def _iter_forward(self, X):
        for i, net in enumerate(self.nets):
            yield net(X[:,i])
        
        
