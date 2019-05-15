import torch, math, numpy

from . import LeNet5, speclib

from .. import modules

class SparseMM(torch.autograd.Function):

    def forward(self, W, X):
        self.save_for_backward(W, X)
        return torch.mm(W, X)

    def backward(self, grad_output):
        W, X = self.saved_tensors
        grad = grad_output.clone()
        dLdW = torch.mm(grad, X.t())
        dLdX = torch.mm(W.t(), grad)
        return dLdW, dLdX

class ChebyshevGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.register_buffer("L", self.scale_laplacian(laplacian).to_dense())
        self.K = K
        #self.weight.data.zero_() this will make it not work

    def scale_laplacian(self, laplacian):
        lmax = speclib.coarsening.lmax_L(laplacian)
        L = speclib.coarsening.rescale_L(laplacian, lmax)

        L = L.tocoo()
        indices = numpy.column_stack((L.row, L.col)).T
        indices = torch.from_numpy(indices).long()
        L_data = torch.from_numpy(L.data).float()

        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        return L

    def forward(self, X):
        N, C, L = X.size()
        X0 = X.permute(1, 2, 0).contiguous().view(C, L*N)
        X1 = torch.mm(self.L, X0)#X1 = SparseMM().forward(self.L, X0)
        Xs = list(self.iter_chebyshev_X(X0, X1))
        out = torch.stack([X0, X1] + Xs, dim=0)
        out = out.view(self.K, C, L, N).permute(3, 1, 2, 0).contiguous()
        out = out.view(N*C, L*self.K)
        return super().forward(out).view(N, C, -1)

    def iter_chebyshev_X(self, X0, X1):
        for k in range(2, self.K):
            X2 = 2 * torch.mm(self.L, X1) - X0#SparseMM().forward(self.L, X1) - X0
            yield X2
            X0, X1 = X1, X2

class ChebyshevGraphConv(torch.nn.Linear):

    def __init__(self, laplacian, K, f1, f2, **kwargs):
        super().__init__(f1//K, f2, bias=False, **kwargs)
        self.register_buffer("L", self.scale_laplacian(laplacian).to_dense())
        self.K = K
        self.act = modules.polynomial.Activation(self.L.size(0), K-1)
        #self.weight.data.zero_() this will make it not work

    def scale_laplacian(self, laplacian):
        lmax = speclib.coarsening.lmax_L(laplacian)
        L = speclib.coarsening.rescale_L(laplacian, lmax)

        L = L.tocoo()
        indices = numpy.column_stack((L.row, L.col)).T
        indices = torch.from_numpy(indices).long()
        L_data = torch.from_numpy(L.data).float()

        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        return L

    def forward(self, X):
        N, C, L = X.size()
        X0 = X.permute(1, 2, 0).contiguous().view(C, L*N)
        X1 = torch.mm(self.L, X0)#X1 = SparseMM().forward(self.L, X0)
        X1 = X1.view(C, L, N).permute(2, 0, 1).contiguous().view(N, C*L)
        #Xs = list(self.iter_chebyshev_X(X0, X1))
        out = self.act(X1).view(N, C, L)#torch.stack([X0, X1] + Xs, dim=0)
        #out = out.view(self.K, C, L, N).permute(3, 1, 2, 0).contiguous()
        #out = out.view(N*C, L*self.K)
        return super().forward(out).view(N, C, -1)

    def iter_chebyshev_X(self, X0, X1):
        for k in range(2, self.K):
            X2 = 2 * torch.mm(self.L, X1) - X0#SparseMM().forward(self.L, X1) - X0
            yield X2
            X0, X1 = X1, X2
class GraphMaxPool(torch.nn.MaxPool1d):

    def forward(self, X):
        X = X.permute(0, 2, 1).contiguous()
        X = super().forward(X)
        return X.permute(0, 2, 1).contiguous()

class LeNet5Graph(torch.nn.Module):

    def __init__(
        self,
        D = 944, 
        cl1_f = 32,
        cl1_k = 25,
        cl2_f = 64,
        cl2_k = 25,
        fc1 = 512,
        fc2 = 10,
        gridsize = 28,
        number_edges = 8,
        coarsening_levels = 4
    ):
        super().__init__()
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
            torch.nn.Dropout(0.5),
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
        conv = ChebyshevGraphConv(laplacian, k, f1, f2)
        return LeNet5.init_module(conv, f1, f2)

    def create_pool(self):
        return GraphMaxPool(4)
