import torch

import src

class BaseModel(torch.nn.Module):

    def __init__(self, D):
        super().__init__()
        self.layer_count = 0
        self.net = torch.nn.Sequential(*self.make_layers(D))

    def count_layers(self):
        return self.layer_count

    def new_layer(self, *args, **kwargs):
        self.layer_count += 1
        return self.create_single_layer(*args, **kwargs)

    def forward(self, X):
        return self.net(X)

    # === ABSTRACT ===

    def make_layers(self, D):
        raise NotImplementedError

    def create_single_layer(self, *args, **kwargs):
        raise NotImplementedError

class BaseResModel(BaseModel):

    # === ABSTRACT ===

    def get_channel_width(self):
        raise NotImplementedError

    # === PROTECTED ===

    def create_resblock(self, C):
        return src.modules.ResBlock(
            block = torch.nn.Sequential(
                self.new_layer(C, C),
                self.new_layer(C, C)
            )
        )

    def make_layers(self, D):
        C = self.get_channel_width()
        return [
            torch.nn.Linear(D, C),

            src.modules.ResNet(
                self.create_resblock(C),
                self.create_resblock(C)
            ),

            torch.nn.Linear(C, 2)
        ]
    
