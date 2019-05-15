import torch

EPS = 1e-32

class ZeroOptimizer(torch.optim.SGD):

    def step(self):
        lr = self.param_groups[0]["lr"]
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad = calc_grad(lr, p, p.grad)
        super().step()

def calc_grad(lr, W, J, eps=EPS):
    Z = calc_z(W, eps)
    G = gravitate_zero(lr, W)
    return Z*G + (1-Z)*J

def gravitate_zero(lr, W):
    G = torch.zeros_like(W)
    A = W.abs()
    I = A > 0
    G[I] = lr/W[I]
    J = (W-lr*G).abs() > A
    G[J] = W[J]/lr
    return G

def calc_p(x, eps=EPS):
    x = x.abs()
    return 1-x/(x.mean()+eps)

def calc_z(w, eps=EPS):
    return torch.nn.functional.relu(calc_p(w, eps))

if __name__ == "__main__":
    
    torch.manual_seed(10)

    w_1 = torch.ones(10)*100-50
    w_r = torch.rand(10)*100-50
    w_s = torch.Tensor([1e-4, -1e-4, 1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6, 1e-8, -1e-8])
    w_w = 10**torch.arange(0, -10, -1).float()
    w_0 = torch.zeros(10)
    w_z = torch.Tensor([1e-4]*3+ [1e-8] + [0]*6)
    w_u = torch.Tensor([1e2] + [1e-10]*9)
    w_v = torch.Tensor([1] + [1e-10])
    w_o = torch.Tensor([1])

    w_x = torch.Tensor([1]*1 + [1e-4]*1)

    def print_wz(w, fmt=".2f"):
        z = calc_z(w)
        buf = "%{}\t%.2f".format(fmt)
        for a, b in zip(w, z):
            print(buf % (a.item(), b.item()))
        input("===")

    #'''
    print_wz(w_1)
    print_wz(w_r)
    print_wz(w_s, fmt=".0E")
    print_wz(w_w, fmt=".0E")
    print_wz(w_0)
    print_wz(w_z, fmt=".0E")
    print_wz(w_u, fmt=".0E")
    print_wz(w_v, fmt=".0E")
    print_wz(w_o)
    #'''

    print_wz(w_x, ".0E")
