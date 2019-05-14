import torch
from torchvision import datasets, transforms

MIU = [0.4914, 0.4824, 0.4467]
STD = [0.2471, 0.2435, 0.2616]

def get(savedir, batchsize, download=0, numworkers=2):
    
    CLASSES = 10
    CHANNELS = 3
    IMAGESIZE = 32

    normalizer = transforms.Normalize(mean=MIU, std=STD)
    
    train = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root = savedir,
            train = True,
            download = download,
            transform = transforms.Compose([
                transforms.RandomCrop(IMAGESIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer
            ])
        ),
        batch_size = batchsize,
        shuffle = True,
        num_workers = numworkers
    )
    
    test = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root = savedir,
            train = False,
            download = False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalizer
            ])
        ),
        batch_size = batchsize*2,
        shuffle = True,
        num_workers = numworkers
    )

    return train, test, CLASSES, CHANNELS, IMAGESIZE
