import torch
from torchvision import datasets, transforms

def get(savedir, batchsize, download=0, numworkers=2):
    
    CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = 28
    
    train = torch.utils.data.DataLoader(
        datasets.MNIST(
            root = savedir,
            train = True,
            download = download,
            transform = transforms.Compose([
                transforms.RandomCrop(size=IMAGESIZE, padding=2),
                transforms.ToTensor()
            ])
        ),
        batch_size = batchsize,
        shuffle = True,
        num_workers = numworkers
    )
    
    test = torch.utils.data.DataLoader(
        datasets.MNIST(
            root = savedir,
            train = False,
            download = False,
            transform = transforms.ToTensor()
        ),
        batch_size = batchsize*2,
        shuffle = True,
        num_workers = numworkers
    )

    return train, test, CLASSES, CHANNELS, IMAGESIZE


