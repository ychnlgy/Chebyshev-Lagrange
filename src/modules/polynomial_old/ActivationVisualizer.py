import torch, numpy, matplotlib
matplotlib.use("agg")

HIST_BINS = 100
N_POINTS = 1000

class ActivationVisualizer(torch.nn.Module):

    AXES = None

    def __init__(self, k, module):
        super().__init__()
        self.k = k
        self.history = []
        self.module = module
        self.axes = None

    def forward(self, X):
        self.history.append(X[:,:self.k].clone().detach().cpu().numpy())
        return self.module(X)

    def reset(self):
        self.history = []

    def visualize(self, figsize, fname, xlim=None):
        history = numpy.concatenate(self.history, axis=0)
        hist, plot = self.get_plots(figsize)
        minmax = []
        for i in range(self.k):
            x = history[:,i].reshape(-1)
            lowhigh = (float(x.min()), float(x.max())) if xlim is None else xlim
            I = (x >= lowhigh[0]) & (x <= lowhigh[1])
            x = x[I]
            self.hist_input(hist[i], x)
            minmax.append(lowhigh)
        y = self.peek_activation(minmax)
        for i in range(self.k):
            self.visualize_activations(
                i,
                plot[i],
                minmax[i],
                y[:,i],
                self.module.basis.nodes.cpu().numpy(),
                self.module.weight[0,i,:].clone().detach().cpu().numpy()
            )
            
        self.finalize_and_save(hist, plot, fname)
        [a.cla() for a in hist]
        [a.cla() for a in plot]

    # === PRIVATE ===

    def finalize_and_save(self, hist, plot, fname):
        plot[self.k//2].legend(bbox_to_anchor=[1.1, -0.1])
        plot[0].set_ylabel("Polynomial output")
        hist[0].set_ylabel("Input count")
        matplotlib.pyplot.savefig(fname, bbox_inches="tight")

    def get_plots(self, figsize):
        if ActivationVisualizer.AXES is None:
            _, ActivationVisualizer.AXES = matplotlib.pyplot.subplots(
                nrows = 2,
                ncols = self.k,
                sharex = "col",
                sharey = "row",
                figsize = figsize
            )
        hist = ActivationVisualizer.AXES[0,:]
        plot = ActivationVisualizer.AXES[1,:]
        return hist, plot

    def peek_activation(self, minmax):
        X = torch.zeros(N_POINTS, self.module.d)
        for i, (xmin, xmax) in enumerate(minmax):
            X[:,i] = torch.linspace(xmin, xmax, N_POINTS)
        with torch.no_grad():
            Y = self.module(X.to(self.module.weight.device))
        return Y.cpu().numpy()

    def hist_input(self, plot, x):
        plot.hist(x, bins=HIST_BINS)

    def visualize_activations(self, i, plot, minmax, y, nodex, nodey):
        plot.plot(numpy.linspace(*minmax, len(y)), y, label="Interpolated polynomial activation")
        plot.plot(nodex, nodey, "x", label="Learned Chebyshev nodes")
        
        plot.axvline(x=nodex[0], linestyle=":", label="Chebyshev x-position")
        for node in nodex[1:]:
            plot.axvline(x=node, linestyle=":")
            
        plot.set_xlabel("$x_%d$" % i)
