import torch, copy

class Savable(torch.nn.Module):

    def __init__(self, module, init_classes):
        super(Savable, self).__init__()
        self.module = module
        self.init_classes = init_classes

    def get_base(self):
        return self.module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def save(self, fname):
        torch.save(self.state_dict(), fname)
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location="cpu"))
    
    def paramcount(self):
        return sum(
            map(
                torch.numel,
                [
                    p for p in self.parameters()
                    if p.requires_grad
                ]
            )
        )

    def clone(self):
        clone = copy.deepcopy(self)
        clone.reinit()
        return clone

    def reinit(self):
        self.apply(self._reinit)

    def _reinit(self, m):
        if type(m) in self.init_classes:
            m.reset_parameters()
