import torch, math

class LeNet5(torch.nn.Module):

    def __init__(
        self,
        node = None
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

        relu = torch.nn.ReLU()

        self.cnn = torch.nn.Sequential(
            LeNet5.create_conv(1, cl1_f, cl1_k),
            relu,
            LeNet5.create_pool(),
            
            self.create_conv(cl1_f, cl2_f, cl2_k),
            relu,
            LeNet5.create_pool(),
        )

        self.net = torch.nn.Sequential(
            LeNet5.create_fc(fc1fin, fc1),
            relu,
            torch.nn.Dropout(0.5),
            LeNet5.create_fc(fc1, fc2)
        )

    def forward(self, X):
        conv = self.cnn(X).view(X.size(0), -1)
        return self.net(conv)

    @staticmethod
    def create_conv(f1, f2, k):
        conv = torch.nn.Conv2d(f1, f2, k, padding=(2, 2))
        fin = f1*k**2
        fout = f2
        return LeNet5.init_module(conv, fin, fout)

    @staticmethod
    def create_pool():
        return torch.nn.MaxPool2d(2, 2)

    @staticmethod
    def create_fc(f1, f2):
        fc = torch.nn.Linear(f1, f2)
        return LeNet5.init_module(fc, f1, f2)

    @staticmethod
    def init_module(mod, fin, fout):
        scale = math.sqrt(2.0/(fin+fout))
        mod.weight.data.uniform_(-scale, scale)
        if mod.bias is not None:
            mod.bias.data.fill_(0)
        return mod
        
