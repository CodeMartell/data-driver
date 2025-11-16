# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:48:17 2023

@author: Edoardo

Unashamedly extracted from the amazing work at:
https://github.com/acfr/LBDN/blob/main/layer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

## from https://github.com/acfr/LBDN/blob/main/layer.py
## a linear layer with Lipschitz upper bound ("scale")
class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale   
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x 
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 * A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x

## from https://github.com/acfr/LBDN/blob/main/layer.py
## a non-linear layer with Lipschitz upper bound ("scale")
## and any activation function ("activation")
## make sure the gradient of the activation does not exceed one
class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, activation=nn.ReLU):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))   
        self.Q = None
        self.act = activation

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # z = sqrt(2) * \Psi^{-1} * B @ h
        if self.bias is not None:
            x += self.bias
        x = self.act(x) * torch.exp(self.psi) # \Psi * activation(z)
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) * A.T @ \Psi * activation(z)
        return x

## copy of model.py with Lipschitz layers instead of generic ones
class DiodeModel(nn.Module):

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
        #act = nn.ELU() # activation function with 1-Lipschitz constant
        act = nn.ReLU()
        gamma = 0.9 # global upper bound on the lipschitz constant
        
        # first layer
        self.layers = []
        self.layers.append(SandwichFc(input_size, hidden_dim, scale=1.0 activation=act)) # Guarantee: Lip(layer) <= 1.0
        
        # intermediate layers
        for i in range(num_layers - 1):
            self.layers.append(SandwichFc(hidden_dim, hidden_dim, scale=1.0, activation=act)) # Guarantee: Lip(layer) <= 1.0
        
        # output layer
        self.layers.append(SandwichLin(hidden_dim, output_size, scale=gamma, AB=True)) # Guarantee: Lip(layer) <= gamma
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
