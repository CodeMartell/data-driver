# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:36:50 2023

@author: Edoardo

Unashamedly extracted from the amazing work at:
https://github.com/araujoalexandre/Lipschitz-SLL-Networks/tree/main/core/models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class SDPBasedLipschitzLinearLayer(nn.Module):
    """
    A layer with guaranteed Lipschitz upper bound ("scale")
    and optional activation function ("activation").
    Make sure the gradient of the activation does not exceed one,
    and use the identity for linear layers.
    """

    def __init__(self, n_hidden, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation

        # manual initialisation of weights and biases
        self.weight = nn.Parameter(torch.empty(n_hidden, n_hidden))
        self.bias = nn.Parameter(torch.empty(n_hidden))
        self.q = nn.Parameter(torch.randn(n_hidden))  # normal distro, mean 0 variance 1

        nn.init.xavier_normal_(self.weight)
        bound = 1 / np.sqrt(n_hidden)
        nn.init.uniform_(self.bias, -bound, bound)

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)  # W @ x + b
        if self.activation:
            y = self.activation(y)  # act(W @ x + b)

        t = self.compute_t()
        y = 2 * F.linear(t * y, self.weight.T)  # 2 * Wt @ T @ act(W @ x + b)

        return x - y  # residual connection


class PaddingChannels(nn.Module):

    def __init__(self, n_in, n_out, mode="zero"):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            n_copy = int(self.n_out / self.n_in)
            return x.repeat(1, n_copy) / np.sqrt(n_copy)
        elif self.mode == "zero":
            y = torch.zeros(x.shape[0], self.n_out, device=x.device)
            y[:, :self.n_in] = x
            return y


class LinearNormalized(nn.Linear):

    def __init__(self, n_in, n_out, bias=True, scale=1.0):
        super().__init__(n_in, n_out, bias)
        self.scale = scale

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(self.scale * x, self.Q, self.bias)


class DiodeModel(nn.Module):
    """
    Copy of model.py with AOL Lipschitz layers instead of generic ones
    """

    def __init__(self,
                 input_size: int = 2,
                 output_size: int = 1,
                 num_layers: int = 4,
                 hidden_dim: int = 8,
                 device: str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # hard-coded parameters (for compatibility)
        act = nn.ReLU()  # gradient-preserving 1-Lipschitz activation
        gamma = 0.99975  # global upper bound on the lipschitz constant

        # first layer
        self.layers = []
        self.layers.append(PaddingChannels(input_size, hidden_dim, mode="clone"))  # Guarantee: Lip(layer) <= 1.0

        # intermediate layers
        for i in range(num_layers):
            self.layers.append(SDPBasedLipschitzLinearLayer(hidden_dim, activation=act))  # Guarantee: Lip(layer) <= 1.0

        # output layer
        self.layers.append(LinearNormalized(hidden_dim, output_size, scale=gamma))  # Guarantee: Lip(layer) <= gamma
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
