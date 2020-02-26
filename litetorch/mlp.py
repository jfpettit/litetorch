import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable


class MLP(nn.Module):
    def __init__(self,
                input_size: int,
                output_size: int,
                hidden_layer_sizes: list,
                activation: Callable = torch.tanh,
                output_activation: Callable = None,
                dropout_layers: list = None,
                dropout_p: float = None,
                out_squeeze: bool = False):
        super(MLP, self).__init__()
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size] 
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.out_squeeze = out_squeeze

        self.dropout_p = dropout_p
        self.dropout_layers = dropout_layers

        for i, l in enumerate(layer_sizes[1:]):
            self.layers.append(nn.Linear(layer_sizes[i], l))

    def forward(self, x: torch.Tensor):
        for l in self.layers[:-1]:
            x = self.activation(l(x))

            if self.dropout_layers is not None and l in self.dropout_layers:
                x = F.dropout(x, p=self.dropout_p)

        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))

        return x.squeeze() if self.out_squeeze else x

