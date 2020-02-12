import torch

def regress2d(x, y):
    '''

    Input:
        x - torch Tensor of shape (d, n), d different sets of n x-points.
        y - torch Tensor of shape (d, n), corresponding y-values.

    Output:
        w - torch Tensor of shape (d, 1), linearly least square
            weights to map x to y.
        b - torch Tensor of shape (d, 1), linearly least square
            bias to map x to y.

    '''
    xm = x.mean(dim=1).unsqueeze(1)
    ym = y.mean(dim=1).unsqueeze(1)
    w = _calc_w(x, y, xm, ym)
    b = _calc_b(x, y, w, xm, ym)
    return w, b

# === PRIVATE ===

def _calc_b(x, y, w, xm, ym):
    return ym-w*xm

def _calc_w(x, y, xm, ym):
    dx = x - xm
    dy = y - ym
    num = (dx*dy).sum(dim=1).unsqueeze(1)
    den = (dx**2).sum(dim=1).unsqueeze(1)
    return num/den

if __name__ == "__main__":

    # Tests for 1d vectors

    wt = [2.5, -0.2]
    bt = [-1.25, 50]
    
    def f(x):
        y = torch.zeros(2, 100)
        y[0] = wt[0]*x +bt[0] + torch.zeros_like(x).normal_(mean=0, std=0.5)
        y[1] = wt[1]*x +bt[1] + torch.zeros_like(x).normal_(mean=0, std=0.5)
        return y

    x = torch.rand(1, 100) * 10 - 5
    y = f(x)

    w, b = regress2d(x, y)

    def similar(v, t, eps=0.1):
        print("Output: %.3f, target: %.3f" % (v, t))
        return abs(v-t) < eps * abs(t)

    for i in range(2):
        assert similar(w[i].item(), wt[i])
        assert similar(b[i].item(), bt[i])
