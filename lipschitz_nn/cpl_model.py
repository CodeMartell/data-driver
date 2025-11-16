"""
Code extracted from:
https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/layers.py

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

import numpy as np

MAX_ITER = 1
EVAL_MAX_ITER = 100


# From https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/util.py
class SpectralNormPowerMethod(nn.Module):

    def __init__(self, input_size, eps=1e-8):
        super(SpectralNormPowerMethod, self).__init__()
        self.input_size = input_size
        self.eps = eps
        self.u = torch.randn(input_size)
        self.u = self.u / self.u.norm(p=2)
        self.u = nn.Parameter(self.u, requires_grad=False)

    def normalize(self, arr):
        norm = torch.sqrt((arr ** 2).sum())
        return arr / (norm + 1e-12)

    def _compute_dense(self, M, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        for _ in range(max_iter):
            v = self.normalize(F.linear(self.u, M))
            self.u.data = self.normalize(F.linear(v, M.T))
        z = F.linear(self.u, M)
        sigma = torch.mul(z, v).sum()
        return sigma

    def _compute_conv(self, kernel, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        pad = (1, 1, 1, 1)
        pad_ = (-1, -1, -1, -1)
        for i in range(max_iter):
            v = self.normalize(F.conv2d(F.pad(self.u, pad), kernel))
            self.u.data = self.normalize(F.pad(F.conv_transpose2d(v, kernel), pad_))
        u_hat, v_hat = self.u, v

        z = F.conv2d(F.pad(u_hat, pad), kernel)
        sigma = torch.mul(z, v_hat).sum()
        return sigma

    def forward(self, M, max_iter):
        """ Return the highest singular value of a matrix
        """
        if len(M.shape) == 4:
            return self._compute_conv(M, max_iter)
        elif len(M.shape) == 2:
            return self._compute_dense(M, max_iter)


# From https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/layers.py
class ConvexPotentialLayerLinear(nn.Module):
    # Modified constructor to take the activation function as input
    def __init__(self, cin, cout, epsilon=1e-4, act=nn.ReLU()):
        super(ConvexPotentialLayerLinear, self).__init__()
        self.activation = act
        self.register_buffer('eval_sv_max', torch.Tensor([0]))

        self.weights = torch.zeros(cout, cin)
        self.bias = torch.zeros(cout)

        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

        self.pm = SpectralNormPowerMethod((1, cin))
        self.train_max_iter = MAX_ITER
        self.eval_max_iter = EVAL_MAX_ITER

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon
        self.alpha = torch.zeros(1)
        self.alpha = nn.Parameter(self.alpha)

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        res = F.linear(res, self.weights.t())
        if self.training == True:
            self.eval_sv_max -= self.eval_sv_max
            sv_max = self.pm(self.weights, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max += self.pm(self.weights, self.eval_max_iter)
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)

        out = x - h * res
        return out


# From https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/layers.py
class LinearNormalized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.scale = scale

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(self.scale * x, self.Q, self.bias)


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


# Copy of model.py with Convex Potential Layers (CPLs)
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
        act = nn.ReLU()
        gamma = 0.99975  # global upper bound on the lipschitz constant

        # first layer
        self.layers = []
        self.layers.append(PaddingChannels(input_size, hidden_dim, mode="clone"))

        # intermediate layers
        for i in range(num_layers):
            self.layers.append(ConvexPotentialLayerLinear(hidden_dim, hidden_dim, act=act))

        # output layer
        self.layers.append(LinearNormalized(hidden_dim, output_size, scale=gamma))
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
