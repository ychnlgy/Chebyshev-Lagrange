import torch

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

from .datasets import add_noise

def visualize(model, outf, X_view, Y_view, testloss, tier, noise):

    with torch.no_grad():

        model.to("cpu")

        yh = model(X_view).mean(dim=-1).numpy()
        y = Y_view.numpy()
        x = X_view[:,0].numpy()
        yn = add_noise(Y_view, noise).numpy()
        
        pyplot.plot(x, yh, ".--", label="Predicted trajectory")
        pyplot.plot(x, y,  label="Ground truth")
        pyplot.plot(x, yn, "x", label="Noisy observations")

        pyplot.legend()

        params = sum(torch.numel(p) for p in model.parameters())

        info = "RMSE: %.5f; Parameters: %d; Layers: %d" % (testloss, params, model.count_layers())
        
        pyplot.title(info)
        pyplot.legend()
        pyplot.savefig(outf)
        print("Saved to \"%s\"." % outf)

        pyplot.clf()
