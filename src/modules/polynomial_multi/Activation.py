import torch, math

from . import chebyshev, LagrangeBasis

class Activation(torch.nn.Module):

    def __init__(self, input_size, n_degree, d_out):
        super().__init__()
        self.d = input_size
        self.n = n_degree + 1
        self.t = d_out
        self.radius = self._calc_radius(self.n)
        self.basis = LagrangeBasis.create(
            chebyshev.get_nodes(self.n, -self.radius, self.radius)
        )
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(1, self.d, self.n, self.t, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0/(self.d+self.n))
        self.weight.data.normal_(mean=0, std=std)

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of shape (N, D, *), input features.

        Output:
            X' - torch Tensor of shape (N, D', *), outputs.

        '''
        N = X.size(0)
        D = X.size(1)
        S = X.shape[2:]
        
        B = self.basis(X.view(N, D, -1)).unsqueeze(3) # (N, D, n, 1, -1)
        L = (self.weight * B).sum(dim=2).sum(dim=1)
        return L.view(N, self.t, *S)

    def _calc_radius(self, n):
        return 1.0/math.cos(math.pi/(2*n))
