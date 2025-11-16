"""
Code taken and adapted from https://github.com/locuslab/orthogonal-convolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, scale=1.0, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.scale = scale
        self.alpha.data = self.weight.norm()

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
        self.Q = None

    def forward(self, X):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        return F.linear(self.scale * X, self.Q if self.training else self.Q.detach(), self.bias)


class GroupSort(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def forward(self, x):
        if not self.k:
            self.k = x.shape[-1]
        y = torch.empty(x.shape)
        for i in range(int(np.ceil(x.shape[-1] / self.k))):
            n = min(self.k * (i + 1), x.shape[-1])  # avoid index overflow
            y[:, self.k * i:n] = torch.sort(x[:, self.k * i:n])[0]
        return y


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


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
        # Gradient Preserving Activation Function (Discontinuous Derivative)
        act = Abs()
        gamma = 0.99975  # global upper bound on the lipschitz constant
        sqrt_g = np.sqrt(gamma)

        # first layer
        self.layers = []
        self.layers.append(CayleyLinear(input_size, hidden_dim, scale=1.0))  # Orthogonal 1-Lipschitz layer :
        self.layers.append(act)                                              # Lip(layer) <= 1.0

        # intermediate layers
        for i in range(num_layers - 1):
            self.layers.append(CayleyLinear(hidden_dim, hidden_dim))  # Orthogonal 1-Lipschitz layer
            self.layers.append(act)                                   # Lip(layer) <= 1.0

        # output layer
        self.layers.append(CayleyLinear(hidden_dim, output_size, scale=gamma))  # Orthogonal 1-Lipschitz layer
        self.model = nn.Sequential(*self.layers).to(device)                     # Lip(layer) <= gamma

    def forward(self, x):
        x = self.model(x)
        return x

