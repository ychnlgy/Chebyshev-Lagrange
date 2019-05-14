import torch, math

from . import speclib

class LeNet5(torch.nn.Module):

    def __init__(
        self,
        nx = 28,
        ny = 28,
        cl1_f = 32,
        cl1_k = 5,
        cl2_f = 64,
        cl2_k = 5,
        fc1 = 512,
        fc2 = 10
    ):
        super().__init__()

        fc1fin = cl2_f*(nx//4)**2

        maxpool = torch.nn.MaxPool2d(2, 2)
        relu = torch.nn.ReLU()

        self.cnn = torch.nn.Sequential(
            self._create_conv(1, cl1_f, cl1_k),
            relu,
            maxpool,
            
            self._create_conv(cl1_f, cl2_f, cl2_k),
            relu,
            maxpool,
        )

        self.net = torch.nn.Sequential(
            self._create_fc(fc1fin, fc1),
            relu,
            torch.nn.Dropout(0.5),
            self._create_fc(fc1, fc2)
        )

    def forward(self, X):
        conv = self.cnn(X)
        print(conv.shape)
        input()
        v = conv.transpose(1, -1).contiguous().view(-1, conv.size(1))
        return self.net(v)

    def _create_conv(self, f1, f2, k):
        conv = torch.nn.Conv2d(f1, f2, k, padding=(2, 2))
        fin = f1*k**2
        fout = f2
        return self._init_module(conv, fin, fout)

    def _create_fc(self, f1, f2):
        fc = torch.nn.Linear(f1, f2)
        return self._init_module(fc, f1, f2)

    def _init_module(self, mod, fin, fout):
        scale = math.sqrt(2.0/(fin+fout))
        mod.weight.data.uniform_(-scale, scale)
        mod.bias.data.fill_(0)
        return mod
        
