import tqdm, torch, sys

from .. import spectral_graph, datasets, util

def main(datadir, graph, node, download=0, device="cuda"):

    '''

    Input:
        datadir - str path to where the dataset files should be stored.
        graph - int or str, represents bool of whether to graph version or not.
        node - int or str, if 0 then use Chebyshev polynomials. If 1 use
            Lagrangian polynomial interpolation of Chebyshev nodes. If -1 use the
            experimental version.
        download - int or str, represents bool of whether the dataset
            should be downloaded if it is not already downloaded.
            Default: 0.
        device - str device to load model and data.

    '''

    download = int(download)
    graph = int(graph)
    node = int(node)

    Model = [spectral_graph.LeNet5, spectral_graph.LeNet5Graph][graph]
    model = Model(node).to(device)

    print("Parameters: %d" % sum(torch.numel(p) for p in model.parameters() if p.requires_grad))

    batchsize = 100
    augment = True

    trainloader, testloader, _, _, _ = datasets.mnist.get(datadir, augment, batchsize, download)

    epochs = 100

    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    trainloss_avg = util.MovingAverage(momentum=0.99)

    for epoch in range(1, epochs+1):

        model.train()
        
        with tqdm.tqdm(trainloader, ncols=80) as bar:
            for x, y in bar:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                loss = lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                trainloss_avg.update(loss.item())

                bar.set_description("[E%d] %.5f" % (epoch, trainloss_avg.peek()))

        sched.step()

        model.eval()

        acc = n = 0.0
        
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                acc += (yh.max(dim=1)[1] == y).float().sum().item()
                n += len(yh)

        sys.stderr.write("Test accuracy: %.2f\n" % (acc/n*100.0))
        
