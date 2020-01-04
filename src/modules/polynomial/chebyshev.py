import torch, math

def get_nodes(n, a, b):
    return chebyshev_node(torch.arange(1, n+1).float(), n, a, b)

# === PRIVATE ===

def chebyshev_node(k, n, a, b):
    return 0.5*(a+b)+0.5*(b-a)*torch.cos((2*k-1)*math.pi/(2*n))
