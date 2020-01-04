import torch

def index_noteye(X):
    '''

    Input:
        X - torch Tensor of size (*, d).

    Output:
        X' - torch Tensor of size (*, d, d-1), where the last column
            contains x[~I], where x is the last column of X.

    '''
    d = X.size(-1)
    T = X.view(-1, d).repeat(1, d)
    I = (1-torch.eye(d)).byte().view(-1).to(X.device)
    return T[:,I].view(*X.size(), d-1)

class LagrangeBasis(torch.nn.Module):

    @staticmethod
    def create(nodes):
        '''

        Description:
            Entry point for instantiating the LagrangeBasis instance.
            We wish to create the polynomial function based on the
            coordinates of the input nodes.

        Input:
            nodes - torch Tensor of shape (n), the x-coordinates of
                the positions we wish to fit an polynomial.
            eps - float, for numerical stability of the denominator.

        Output:
            p - LagrangeBasis, the n-basis functions that will create
                a polynomial that intersects each x-coordinate exactly.

        '''
        n = len(nodes)

        # xm is shape (1, 1, n, n-1, 1)
        #   - index[2]: number of basis functions.
        #   - index[3]: points involved per basis function.
        xm = index_noteye(nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        xj = nodes.view(1, n, 1, 1)

        # dn is shape (1, 1, n, 1)
        denominator = (xj-xm).prod(dim=3)
        return LagrangeBasis(nodes, xm, denominator)

    def forward(self, X):

        '''

        Input:
            X - torch Tensor of size (N, D, *)

        Output:
            L - torch Tensor of size (N, D, n, *), where n is the
                number of nodes for the Lagrange basis.

        '''
        N, D, shape, X = self._prep_shape(X)
        out = (X - self.xm).prod(dim=3)/self.dn
        return out.view(N, D, self.n, *shape)

    def grad_pos1(self):
        return self._calc_grad(1)

    def grad_neg1(self):
        return self._calc_grad(-1)

    # === PROTECTED ===

    def __init__(self, nodes, xm, denominator):
        super().__init__()
        self.register_buffer("nodes", nodes)
        self.nodes.requires_grad = False
        self.register_buffer("xm", xm)
        self.xm.requires_grad = False
        self.register_buffer("dn", denominator)
        self.dn.requires_grad = False
        self.n = self.dn.size(2)

    # === PRIVATE ===

    def _prep_shape(self, X):
        N = X.size(0)
        D = X.size(1)
        shape = X.shape[2:] # for restoring size at the end
        
        X = X.view(N, D, 1, 1, -1)
        return N, D, shape, X

    def _calc_grad(self, x):
        xm = index_noteye(self.xm.squeeze(-1))
        return ((x-xm).prod(-1)/self.dn).sum(dim=-1).squeeze()
    
if __name__ == "__main__":

    def equals_float(v, w):
        eps = 1e-6
        return (v-w).norm() < eps

    # Test [~I]
    x = torch.Tensor([1, 2, 3, 4])
    xp = index_noteye(x)
    assert equals_float(xp, torch.Tensor([
        [2, 3, 4],
        [1, 3, 4],
        [1, 2, 4],
        [1, 2, 3]
    ]))

    # Test: {x1 = -1, x2 = 0, x3 = 1}
    nodes = torch.Tensor([-1, 0, 1])
    basis = LagrangeBasis.create(nodes)

    # We expect:
    #   l1(x) = x(x-1)/2
    #   l2(x) = (x+1)(x-1)/-1
    #   l3(x) = (x+1)x/2

    assert equals_float(basis.xm, torch.FloatTensor([
        [ 0, 1],
        [-1, 1],
        [-1, 0]
    ]).view(1, 1, 3, 2, 1)) # xm is what we subtract

    assert equals_float(basis.dn, torch.FloatTensor([
        [2, -1, 2]
    ]).view(1, 1, 3, 1)) # dn is what we divide

    assert equals_float(basis.grad_pos1(), torch.FloatTensor([
        0.5, -2, 1.5
    ]))

    assert equals_float(basis.grad_neg1(), torch.FloatTensor([
        -1.5, 2, -0.5
    ])) 
