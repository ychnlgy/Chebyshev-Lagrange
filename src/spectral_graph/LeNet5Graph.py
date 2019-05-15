import torch, math, numpy

from . import LeNet5, speclib

from .. import modules

class SparseMM(torch.autograd.Function):

    def forward(self, W, X):
        tosave = [t if t.requires_grad else None for t in [W, X]]
        self.save_for_backward(*tosave)
        return torch.mm(W, X)

    def backward(self, grad_output):
        W, X = self.saved_tensors
        grad = grad_output.clone()
        dLdW = torch.mm(grad, X.t()) if X is not None else None
        dLdX = torch.mm(W.t(), grad) if W is not None else None
        return dLdW, dLdX

class PolyGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, d_in, d_out, **kwargs):
        super().__init__(d_in, d_out, bias=False, **kwargs)
        self.register_buffer("L", self.scale_laplacian(laplacian))
        self.L.requires_grad = False
        self.K = K
        #self.poly = modules.polynomial.LinkActivation(2, d_in, n_degree=3, zeros=False)
        #self.weight.data.zero_() this will make it not work

    def scale_laplacian(self, L):
        #print(L.max(), L.min())
        lmax = speclib.coarsening.lmax_L(L)
        L = speclib.coarsening.rescale_L(L, lmax)

        #print(L.max(), L.min())

        L = L.tocoo()
        indices = numpy.column_stack((L.row, L.col)).T
        indices = torch.from_numpy(indices).long()
        L_data = torch.from_numpy(L.data).float()

        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        return L

    def forward(self, X):
        N, C, L = X.size()
        X0 = X.permute(1, 2, 0).contiguous().view(C, L*N)
        X1 = SparseMM().forward(self.L, X0)
        Xs = list(self.iter_chebyshev_X(X0, X1))
        out = torch.stack([X0, X1] + Xs, dim=0)
        out = out.view(self.K, C, L, N).permute(3, 1, 2, 0).contiguous()
        out = out.view(N*C, L*self.K)
        #out = self.poly(out)
        return super().forward(out).view(N, C, -1)

    def iter_chebyshev_X(self, X0, X1):
        for k in range(2, self.K):
            X2 = 2 * SparseMM().forward(self.L, X1) - X0
            yield X2
            X0, X1 = X1, X2

class NodeGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, d_in, d_out, **kwargs):
        super().__init__(d_in, d_out, bias=False, **kwargs)
        self.register_buffer("L", self.scale_laplacian(laplacian))
        self.K = K
        self.dout = d_out
        values = self.L._values()
        self.act = modules.polynomial.LagrangeBasis.create(
            modules.polynomial.chebyshev.get_nodes(K, values.min(), values.max())
        )
        print(values.min(), values.max())

    def scale_laplacian(self, L):
        lmax = speclib.coarsening.lmax_L(L)
        L = speclib.coarsening.rescale_L(L, lmax)

        L = L.tocoo()
        
        indices = numpy.column_stack((L.row, L.col)).T
        indices = torch.from_numpy(indices).long()
        L_data = torch.from_numpy(L.data).float()

        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        return L.coalesce()

    def forward(self, X):
        N, C, L = X.size()
        
        device = self.L.device
        L_i = self.L._indices()
        L_v = self.L._values()
        
        pL_i = L_i.repeat(1, self.K)
        pL_k = torch.arange(self.K).view(-1, 1).repeat(1, L_i.size(1)).view(-1).long()
        assert pL_i.size(1) == pL_k.size(0)
        pL_i[0] += pL_k.to(device) * self.L.size(0)
        
        pL_v = self.act(self.L._values().unsqueeze(0)).view(-1) # 1, n_laplacian, K
        pL = torch.cuda.sparse.FloatTensor(pL_i, pL_v, torch.Size([self.K*C, C]))
        
        X0 = X.permute(1, 2, 0).contiguous().view(C, L*N)
        out = SparseMM().forward(pL, X0) # K*C, L*N
        out = out.view(self.K, C, L, N).transpose(0, -1).contiguous().view(N*C, L*self.K)
        return super().forward(out).view(N, C, -1)

class ReLUGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, d_in, d_out, **kwargs):
        super().__init__(d_in//K, d_out, bias=False, **kwargs)

    def forward(self, X):
        N, C, L = X.size()
        return torch.nn.functional.relu(super().forward(X.view(N*C, L))).view(N, C, -1)

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

import os

class ExptGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, d_in, d_out, **kwargs):
        super().__init__(d_in//K, d_out, bias=False, **kwargs)
        self.register_buffer("L", self.scale_laplacian(laplacian))
        self.K = K
        self.dout = d_out
        values = self.L._values()
        self.cut = None
        self.act = modules.polynomial.RegActivation(K//2, d_in//K, n_degree=K-1, d_out=d_out)

    def scale_laplacian(self, L):
        #lmax = speclib.coarsening.lmax_L(L)
        #L = speclib.coarsening.rescale_L(L, lmax)
        
        L = L.tocoo()
        
        indices = numpy.column_stack((L.row, L.col)).T
        indices = torch.from_numpy(indices).long()
        L_data = torch.from_numpy(L.data).float()

        pyplot.hist(L.data, bins=100)

        i = 0
        fname = lambda j: "laplacians-%d.png" % j
        while os.path.isfile(fname(i)):
            i += 1
        pyplot.savefig(fname(i))
        pyplot.clf()

        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        return L.coalesce()

    def forward(self, X):
        N, C, L = X.size()

        #pL = self.L.clone()
        #pL._values()[:] = self.act(self.L._values().unsqueeze(0)).view(-1)
        
        X0 = X.permute(1, 2, 0).contiguous().view(C, L*N)
        out = SparseMM().forward(self.L, X0) # C, L*N
        out = out.view(C, L, N).permute(2, 0, 1).contiguous().view(N*C, L)
        out = self.act(out).view(N, C, -1)
        return out#super().forward(out).view(N, C, -1)

class GraphMaxPool(torch.nn.MaxPool1d):

    def forward(self, X):
        X = X.permute(0, 2, 1).contiguous()
        X = super().forward(X)
        return X.permute(0, 2, 1).contiguous()

class LeNet5Graph(torch.nn.Module):

    def __init__(
        self,
        node,
        D = 944, 
        cl1_f = 32,
        cl1_k = 6,
        cl2_f = 64,
        cl2_k = 6,
        fc1 = 512,
        fc2 = 10,
        gridsize = 28,
        number_edges = 8,
        coarsening_levels = 4
    ):
        super().__init__()
        self.Conv = [PolyGraphConv, NodeGraphConv, ReLUGraphConv, ExptGraphConv][node]
        
        L, self.perm = self.generate_laplacian(gridsize, number_edges, coarsening_levels)
        
        fc1fin = cl2_f*(D//16)

        relu = torch.nn.ReLU()

        self.cnn = torch.nn.Sequential(
            self.create_conv(cl1_k, cl1_f, cl1_k, L[0]),
            relu,
            self.create_pool(),

            self.create_conv(cl2_k*cl1_f, cl2_f, cl2_k, L[2]),
            relu,
            self.create_pool(),
        )

        self.net = torch.nn.Sequential(
            LeNet5.create_fc(fc1fin, fc1),
            relu,
            torch.nn.Dropout(0.2),
            LeNet5.create_fc(fc1, fc2)
        )

    def transform_data(self, X):
        return torch.from_numpy(
            speclib.coarsening.perm_data(X.cpu().numpy(), self.perm)
        ).float().to(X.device)
            
    def forward(self, X):
        N, C, W, H = X.size()
        assert C == 1
        X = self.transform_data(X.view(N, W*H)).unsqueeze(-1)
        conv = self.cnn(X).view(X.size(0), -1)
        return self.net(conv)

    # === PROTECTED ===

    def generate_laplacian(self, gridsize, number_edges, coarsening_levels):
        A = speclib.grid_graph.grid_graph(gridsize, number_edges)
        return speclib.coarsening.coarsen(A, coarsening_levels)

    def create_conv(self, f1, f2, k, laplacian):
        conv = self.Conv(laplacian, k, f1, f2)
        return LeNet5.init_module(conv, f1, f2)

    def create_pool(self):
        return GraphMaxPool(4)
