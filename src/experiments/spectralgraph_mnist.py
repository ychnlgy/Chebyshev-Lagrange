import tqdm, torch

from .. import spectral_graph, datasets, util

def main(datadir, download=0):

    '''

    Input:
        datadir - str path to where the MNIST dataset should be stored.
        download - int or str, represents bool of whether the dataset
            should be downloaded if it is not already downloaded.
            Default: 0.

    '''

    download = int(download)

    model = spectral_graph.LeNet5()

    batchsize = 100

    trainloader, testloader, _, _, _ = datasets.mnist.get(datadir, batchsize, download)

    epochs = 20

    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    trainloss_avg = util.MovingAverage(momentum=0.99)

    for epoch in range(1, epochs+1):

        model.train()
        
        with tqdm.tqdm(trainloader, ncols=80) as bar:
            for x, y in bar:
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
                yh = model(x)
                acc += (yh.max(dim=1)[1] == y).float().sum().item()
                n += len(yh)

        sys.stderr.write("Test accuracy: %.3f\n" % (acc/n))
        
