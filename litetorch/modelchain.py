import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable

class ModelChain(nn.Module):
    def __init__(self, model_list, output_squeeze=False):
        self.model_list = model_list
        self.output_squeeze = output_squeeze

    def forward(self, x):
        for model in self.model_list:
            x = model(x)

        return x.squeeze() if self.output_squeeze else x