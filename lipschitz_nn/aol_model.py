# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:50 2023

@author: Edoardo

Unashamedly extracted from the amazing work at:
https://github.com/berndprach/AOL/tree/main/src/aol_code/layers/aol
originally written in Tensorflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_rescaling_factors(parameter_matrix, epsilon=1e-6):
    """
    Computes the diagonal entries of matrix D = (sum_col|Pt * P| ** (-1/2))
    """
    PtP = parameter_matrix.transpose(0, 1) @ parameter_matrix
    PtP_abs = torch.abs(PtP)
    lipschitz_bounds_squared = torch.sum(PtP_abs, dim=1)
    rescaling_factors = (lipschitz_bounds_squared + epsilon) ** (-1 / 2)

    return rescaling_factors


class AOLDense(nn.Module):
    """
    A layer with guaranteed Lipschitz upper bound ("scale")
    and optional activation function ("activation").
    Make sure the gradient of the activation does not exceed one,
    and use the identity for linear layers.
    """

    def __init__(self, n_in, n_out, bias=True, scale=1.0, activation=nn.ReLU()):
        super().__init__()
        self.scale = scale
        self.activation = activation

        # manual initialisation of weights and biases
        self.weights = nn.Parameter(torch.empty(n_out, n_in))
        self.bias = nn.Parameter(torch.empty(n_out))

        nn.init.xavier_normal_(self.weights)
        bound = 1 / np.sqrt(n_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        D = get_rescaling_factors(self.weights)
        res = F.linear(self.scale * D * x, self.weights, self.bias)  # PD @ x * scale
        return self.activation(res) if self.activation else res


def GroupSort(x, k=None):
    if not k:
        k = x.shape[-1]

    y = torch.empty(x.shape)
    for i in range(int(np.ceil(x.shape[-1] / k))):
        n = min(k * (i + 1), x.shape[-1])  # avoid index overflow
        y[:, k*i:n] = torch.sort(x[:, k*i:n])[0]

    return y


class DiodeModel(nn.Module):
    """
    Copy of model.py with AOL Lipschitz layers instead of generic ones
    """

    def __init__(self,
                 input_size: int = 2,
                 output_size: int = 1,
                 num_layers: int = 4,
                 hidden_dim: int = 8,
                 device: str = 'cpu',
                 gamma = 0.99975):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # hard-coded parameters (for compatibility)
        act = torch.abs

        # first layer
        self.layers = []
        self.layers.append(AOLDense(input_size, hidden_dim, scale=1.0, activation=act))  # Lip(layer) <= 1.0

        # intermediate layers
        for i in range(num_layers - 1):
            self.layers.append(AOLDense(hidden_dim, hidden_dim, scale=1.0, activation=act))  # Lip(layer) <= 1.0

        # output layer
        self.layers.append(AOLDense(hidden_dim, output_size, scale=gamma, activation=None))  # Lip(layer) <= gamma
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
