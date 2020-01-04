import torch, math

from . import chebyshev, LagrangeBasis

class Activation(torch.nn.Module):

    def __init__(self, input_size, n_degree, zeros=False, a=None, b=None):
        super().__init__()
        self.d = input_size
        self.n = n_degree + 1

        if a is None:
            a = -self._calc_radius(self.n)
            b = -a
        self.basis = LagrangeBasis.create(
            chebyshev.get_nodes(self.n, a, b)
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(1, self.d, self.n, 1)
        )

        if not zeros:
            self.randomize_parameters()

    def randomize_parameters(self):
        scale = math.sqrt(2.0/(self.d+self.n))
        self.weight.data.uniform_(-scale, scale)

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of shape (N, D, *), input features.

        Output:
            X' - torch Tensor of shape (N, D, *), outputs.

        '''
        N = X.size(0)
        D = X.size(1)
        
        B = self.basis(X.view(N, D, -1)) # (N, D, n, -1)
        L = (self.weight * B).sum(dim=2)
        return L.view(X.size())

    def _calc_radius(self, n):
        return 1.0/math.cos(math.pi/(2*n))
