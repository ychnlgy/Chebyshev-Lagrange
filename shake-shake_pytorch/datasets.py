# -*- coding: utf-8 -*-

import torch

from torchvision import datasets
from torchvision import transforms


def fetch_bylabel(label):
    if label == 10:
        normalizer = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                          std=[0.2471, 0.2435, 0.2616])
        data_cls = datasets.CIFAR10
    else:
        normalizer = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                          std=[0.2675, 0.2565, 0.2761])
        data_cls = datasets.CIFAR100
    return normalizer, data_cls

def load_mnist(batch_size):

    channels = 1
    
    train = datasets.MNIST(root="./data/mnist", train=True, download=1,
                                       transform=transforms.Compose([
                                           transforms.Pad(padding=2),
                                           transforms.RandomCrop(size=32, padding=2),
                                           transforms.ToTensor()
                                        ])
    )
    
    test = datasets.MNIST(root="./data/mnist", train=False, download=0,
                                      transform=transforms.Compose([
                                           transforms.Pad(padding=2),
                                           transforms.ToTensor(),
                                        ])
    )

    train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train, test, channels

def load_dataset(label, batch_size, mnist):

    if mnist:
        return load_mnist(batch_size)
    
    channels = 3

    normalizer, data_cls = fetch_bylabel(label)

    train_loader = torch.utils.data.DataLoader(
        data_cls("./data/cifar{}".format(label), train=True, download=True,
                 transform=transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     normalizer
                 ])),
        batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        data_cls("./data/cifar{}".format(label), train=False, download=False,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     normalizer
                 ])),
        batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, channels
