import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable

from litetorch.utils import conv2d_output_size, conv2d_output_shape

class CNN(nn.Module):
    def __init__(self, 
                kernel_size: int,
                stride: int,
                output_size: int,
                input_channels: int,
                input_height: int,
                channels: list = [64, 64],
                linear_layer_sizes: list = [512],
                activation: Callable = torch.relu,
                output_activation: Callable = None,
                dropout_layers: list = None,
                dropout_p: float = None,
                out_squeeze: bool = False):

        super(CNN, self).__init__()

        conv_sizes = [input_channels] + channels
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.out_squeeze = out_squeeze

        self.dropout_p = dropout_p
        self.dropout_layers = dropout_layers

        self.hw=input_height
        for i, l in enumerate(conv_sizes[1:]):
            self.hw = conv2d_output_size(kernel_size=kernel_size, stride=stride, sidesize=self.hw)
            self.layers.append(nn.Conv2d(conv_sizes[i], l, kernel_size=kernel_size, stride=stride))

        self.hw = (self.hw, self.hw)
        conv_out_size = 1
        for num in self.hw:
            conv_out_size *= num
        conv_out_size *= conv_sizes[-1]

        linear_sizes = [conv_out_size] + linear_layer_sizes + [output_size]
        self.layers.append(nn.Flatten())
        for i, l in enumerate(linear_sizes[1:]):
            self.layers.append(nn.Linear(linear_sizes[i], l))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            print(l)

            if self.dropout_layers is not None and l in self.dropout_layers:
                x = F.dropout(x, p=self.dropout_p)

        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))

        return x.squeeze() if self.out_squeeze else x
