import torch

from . import model
import src

class BaseModel(model.BaseModel):

    def create_single_layer(self, input_size, output_size):
        return torch.nn.Linear(input_size, output_size)

class BaseResModel(model.BaseResModel):

    def create_single_layer(self, input_size, output_size):
        return BaseModel.create_single_layer(self, input_size, output_size)

# === Standard layer-by-layer networks ===          
                        
class Tier_polyproto(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.PrototypeSimilarity(d, d),
                        src.modules.polynomial.Activation(d, 3),
                        self.new_layer(d, d)
                    )
                ) for i in range(3)]
            ),
            self.new_layer(d, 1)
        ]

class Tier_tanhpoly(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.Tanh(),
                        src.modules.polynomial.Activation(d, 3),
                        self.new_layer(d, d)
                    )
                ) for i in range(3)]
            ),
            self.new_layer(d, 1)
        ]

class Tier_chebypoly(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.polynomial.ChebyshevActivation(d, 3),
                        self.new_layer(d, d)
                    )
                ) for i in range(3)]
            ),
            self.new_layer(d, 1)
        ]

class Tier_actpoly(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.polynomial.Activation(d, 3),
                        self.new_layer(d, d)
                    )
                ) for i in range(3)]
            ),
            self.new_layer(d, 1)
        ]

class Tier_cubic(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.Operation(lambda X: X**3),
                        self.new_layer(d, d),
                    )
                ) for i in range(3)]
            ),
            src.modules.Operation(lambda X: X**3),
            self.new_layer(d, 1)
        ]

class Tier_linkpoly(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.polynomial.LinkActivation(2, d, 3),
                        self.new_layer(d, d),
                    )
                ) for i in range(3)]
            ),
            src.modules.polynomial.LinkActivation(2, d, 3),
            self.new_layer(d, 1)
        ]

class Tier_regpoly(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.polynomial.RegActivation(2, d, 3),
                        self.new_layer(d, d),
                    )
                ) for i in range(3)]
            ),
            src.modules.polynomial.RegActivation(2, d, 3),
            self.new_layer(d, 1)
        ]

class Tier_relu(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.ReLU(),
                        self.new_layer(d, d),
                    )
                ) for i in range(3)]
            ),
            torch.nn.ReLU(),
            self.new_layer(d, 1)
        ]

class Tier_tanh(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.Tanh(),
                        self.new_layer(d, d)
                    )
                ) for i in range(3)]
            ),
            torch.nn.Tanh(),
            self.new_layer(d, 1)
        ]

class Tier_relurelu(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.ReLU(),
                        self.new_layer(d, d),

                        torch.nn.ReLU(),
                        self.new_layer(d, d),
                    )
                ) for i in range(3)]
            ),
            torch.nn.ReLU(),
            self.new_layer(d, 1)
        ]

class Tier_relureludeep(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.ReLU(),
                        self.new_layer(d, d),

                        torch.nn.ReLU(),
                        self.new_layer(d, d),
                    )
                ) for i in range(6)]
            ),
            torch.nn.ReLU(),
            self.new_layer(d, 1)
        ]

class Tier_reludeep(BaseModel):

    def make_layers(self, D):
        d = 32
        return [
            self.new_layer(D, d),
            src.modules.ResNet(
                *[src.modules.ResBlock(
                    torch.nn.Sequential(
                        torch.nn.ReLU(),
                        self.new_layer(d, d),
                    )
                ) for i in range(6)]
            ),
            torch.nn.ReLU(),
            self.new_layer(d, 1)
        ]
