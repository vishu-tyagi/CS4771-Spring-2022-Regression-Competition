from typing import Any

import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(
        self,
        dimensions: list[int],
        dropout: int,
        **kw: dict[str, Any]
    ):
        super(MLPModel, self).__init__()
        self.input_dim = dimensions[0]
        self.output_dim = dimensions[-1]
        self.hidden_dims = dimensions[1:-1]
        self.layers = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers_init()

    def layers_init(self):
        self.layers = nn.ModuleList()
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


