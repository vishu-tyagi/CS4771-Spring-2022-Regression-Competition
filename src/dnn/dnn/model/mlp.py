import torch.nn as nn

from dnn.config import DNNConfig


class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: DNNConfig
    ):
        super(MLPModel, self).__init__()
        self.name = config.MODEL_NAME
        self.input_dim = input_dim
        self.output_dim = config.output_dim
        self.hidden_dims = config.MODEL_HIDDEN_DIMENSIONS
        self.layers = nn.ModuleList()

        # Model layers
        self.layers_init()
        # Activation
        self.relu = nn.ReLU()
        # Dropout
        self.dropout = nn.Dropout(config.MODEL_DROPOUT_SIZE)

    def layers_init(self):
        current_dim = self.input_dim
        for hdim in self.hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_dim))
        return self

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        out = self.layers[-1](x)
        return out
