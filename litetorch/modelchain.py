import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable

class ModelChain(nn.Module):
    def __init__(self, model_list, output_squeeze=False):
        """
        Daisy-chain a list of models together. 

        :param model_list: list of models to chain
        :param output_squeeze: whether or not to squeeze output

        This class takes in a list of PyTorch modules and runs a forward pass through each of them, returning the output of the final module.
        It is the user's responsibility to make sure all input/output dimensions between modules match.
        The modules should be initialized before being fed to ModelChain.

        Example:
            mlp1 = MLP(input_size=10, output_size=2)
            mlp2 = MLP(input_size=2, output_size=10)
            chained = ModelChain([mlp1, mlp2])

            x = torch.randn(5, 10)
            out = chained(x)

        """
        super(ModelChain, self).__init__()
        self.model_list = nn.ModuleList(model_list)
        self.output_squeeze = output_squeeze

    def forward(self, x):
        for model in self.model_list:
            x = model(x)

        return x.squeeze() if self.output_squeeze else x