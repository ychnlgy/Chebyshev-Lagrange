import torch, math

from . import Activation
from .. import tensortools

class RegActivation(Activation):

    def __init__(self, n_regress, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leftslice = slice(-n_regress, None)
        self.rightslice = slice(n_regress)

    def forward(self, X_in):
        '''

        Input:
            X - torch Tensor of shape (N, D, *), input features.

        Output:
            X' - torch Tensor of shape (N, D', *), outputs.

        '''
        N = X_in.size(0)
        D = X_in.size(1)
        
        X = X_in.view(N, D, -1)
        
        regress_l = self._regress(self.leftslice, -1) # left or <-1
        regress_r = self._regress(self.rightslice, 0) # right or >1
        
        requires_regress_l = (X < -1).unsqueeze(2) # (N, D, 1, -1)
        requires_regress_r = (X > +1).unsqueeze(2)        

        input("A")
        torch.cuda.empty_cache()
        
        B = self.basis(X).unsqueeze(3) # (N, D, n, 1, -1)
        L = (self.weight * B).sum(dim=2) # (N, D, D', -1)

        input("B")
        torch.cuda.empty_cache()

        dl = self._do_regress(X, *regress_l)
        L[requires_regress_l.expand_as(L)] = dl[requires_regress_l.expand_as(dl)]
        dr = self._do_regress(X, *regress_r)
        L[requires_regress_r.expand_as(L)] = dr[requires_regress_r.expand_as(dr)]

        input("C")
        torch.cuda.empty_cache()
        
        return L.sum(dim=1).view(N, self.t, *X_in.shape[2:])

    # === PROTECTED ===

    def calc_weight(self, slc, x, y):
        "Returns tensor of shape (D, 1)."
        w, _ = tensortools.regress2d(x, y) # D, 1
        return w

    # === PRIVATE ===

    def _do_regress(self, X, w, b):
        e = len(X.shape) - len(w.shape)
        X = X.unsqueeze(2)
        D = X.size(1)
        w = w.view(1, D, -1, *([1]*e))
        b = b.view(1, D, -1, *([1]*e))
        return X*w + b

    def _regress(self, slc, endi):
        x = self.basis.nodes[slc].unsqueeze(0) # 1, d
        y = self.weight[0,:,slc,:,0].transpose(-2, -1).contiguous()
        assert len(y.shape) == 3
        y = y.view(-1, y.size(-1))
        w = self.calc_weight(slc, x, y)

        # we want the discontinuous function to
        # still appear as continuous as possible,
        # so we connect the linear regression to
        # the last point at which the polynomial stops.
        b = y[:,endi].unsqueeze(-1)-w*x[:,endi].unsqueeze(-1)
        assert w.size() == b.size()
        return w, b
