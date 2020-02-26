import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable

class RNN(nn.Module):
    def __init__(self, 
                hidden_size: int, 
                embed_size: int,
                vocab_size: int,
                embed: bool = True,
                num_rnn_layers: int = 1,
                output_size: int = None,
                linear_layer_sizes: list = None,
                activation: Callable = None,
                output_activation: Callable = None,
                dropout_layers: list = None,
                dropout_p: float = None,
                out_squeeze: bool = False):

        super(RNN, self).__init__()

        self.embed = embed
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.linear_layers = None
        self.out_squeeze = out_squeeze

        self.dropout_p = dropout_p
        self.dropout_layers = dropout_layers

        if embed is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_rnn_layers)

        if linear_layer_sizes is not None:
            linear_layer_sizes = [hidden_size] + linear_layer_sizes + [output_size]

            self.linear_layers = nn.ModuleList()
            for i, l in enumerate(linear_layer_sizes[1:]):
                self.linear_layers.append(nn.Linear(linear_layer_sizes[i], l))

    def forward(self, x, hid=None):
        if self.embed:
            x = self.embedding(x)
        x, hid = self.rnn(x, hid)
        
        if self.linear_layers is not None:
            hid = hid.squeeze(0)
            for l in self.linear_layers[:-1]:
                if self.activation is not None:
                    hid = self.activation(l(hid))
                else:
                    hid = l(hid)

            if self.dropout_layers is not None and l in self.dropout_layers:
                hid = F.dropout(hid, p=self.dropout_p)

            if self.output_activation is None:
                hid = self.linear_layers[-1](hid)
            else:
                hid = self.output_activation(self.linear_layers[-1](hid))

        return x.squeeze(), hid if self.out_squeeze else x, hid