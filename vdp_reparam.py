# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:16:53 2024

@author: Edoardo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from lipschitz_nn.aol_model import AOLDense
from error_bounds import pred_error_norm, empirical_lipschitz

"""
Lipschitz-bounded neural network based on approximately orthogonal layer (AOL)
"""
class LipNN(torch.nn.Module):

    def __init__(self,
                 input_size = 2,
                 output_size = 2,
                 num_layers = 4,
                 hidden_dim = 16,
                 act_func = torch.abs,
                 lip_bound = 1.0,
                 device = 'cpu'):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.act_func = act_func
        self.lip_bound = lip_bound
        self.device = device

        # first layer
        self.layers = []
        self.layers.append(AOLDense(input_size, hidden_dim, scale=1.0, activation=act_func))  # Lip(layer) <= 1.0

        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(AOLDense(hidden_dim, hidden_dim, scale=1.0, activation=act_func))  # Lip(layer) <= 1.0

        # output layer (with reparametrisation)
        output_coeffs = output_size * input_size
        self.layers.append(AOLDense(hidden_dim, output_coeffs, scale=lip_bound, activation=None))  # Lip(layer) <= gamma
        self.model = torch.nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        
        g = self.model(x)
        
        # reshape input and network output as matrices
        G = g.view(g.shape[:-1] + (self.output_size, self.input_size))
        X = x.view(x.shape + (1,))

        # reparametrisation: y = g(x) @ x
        Y = torch.bmm(G, X)
        
        # remove last unitary dimension
        y = Y.view(Y.shape[:-1])

        return y

"""
Root mean squared error divided by input norm
"""
class CustomLoss(torch.nn.Module):
    
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, outputs, targets, epsilon=1e-2): # smaller eps creates training instability :-(
        
        squared_error = torch.norm(targets - outputs, dim=-1)
        input_norm = torch.norm(inputs, dim=-1)
        
        loss = squared_error / (input_norm + epsilon)
        return loss.mean()

"""
Runs one epoch of training

Parameters:
    model -> neural network to be trained
    device -> computational device in use, e.g. "cpu"
    loader -> DataLoader providing the data in batches
    loss_func -> torch function that computes the loss between output and target
    optimizer -> torch.optim object to update the model parameters
    epoch -> epoch number for displaying progress only
    verbosity -> roughly how many updates per epoch to print on screen
"""
def train(model, device, loader, loss_func, optimizer, epoch, verbosity=2):
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad() # reset gradients
        outputs = model(inputs) # forward pass
        loss = loss_func(inputs, outputs, targets) # loss
        loss.backward() # backward pass
        optimizer.step() # update parameters
        
        if (batch_idx + 1) % round(len(loader) / verbosity) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(loader.dataset),
                100. * batch_idx / len(loader), loss))

# load the downsampled dataset
data = np.loadtxt("./data/vdp_clean.csv", delimiter=",").astype(np.float32)

# create a dataset with explicit input-output tuples
dataset = [(data[i,:2], data[i,2:]) for i in range(data.shape[0])]

# hard-coded training parameters
seed = 12345
torch.manual_seed(seed)
device = torch.device("cpu")
batch_size = 64
learn_rate = 1e-4
epochs = 50
#loss_func = torch.nn.functional.mse_loss
loss_func = CustomLoss()

# split the training set into batches
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

# create the neural network model and associated optimiser
lip_bound = 0.1
model = LipNN(num_layers = 4,
              hidden_dim = 8,
              act_func = torch.abs,
              lip_bound = lip_bound,
              device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# training!
for epoch in range(epochs):
    train(model, device, loader, loss_func, optimizer, epoch + 1)

# save trained model
torch.save(model.state_dict(), "vdp_aol.pt")

# find largest hole in the training set
plt.figure()
plt.plot(data[:,0], data[:,1], ".")

# dataset grid margin
margin = (0.1 / 2) * np.sqrt(2)
print("Margin:", margin)

# compute largest training error in Euclidean norm
print("Computing worst-case prediction error on training set")

# compute the norm of the prediction error
pen = pred_error_norm(model, device, loader)

# sort the prediction error according to the input norm
sid = np.argsort(pen[:,0])
pen = pen[sid]

# force monotonicity by copying the previous largest entry
# skip entries close to zero (numerical instability)
eps = 1e-12
pen_bound = pen[pen[:,0] >= eps].copy()
pen_bound[:,1] = np.maximum.accumulate(pen_bound[:,1])

# plot
plt.figure()
plt.plot(pen[:,0], pen[:,1], ".r", label="raw")
plt.plot(pen_bound[:,0], pen_bound[:,1], "k", label="max")
plt.xlabel("Input Norm ||x||")
plt.ylabel("Training Error ||f(x)-f'(x)||")
plt.legend(loc="upper left")

# repeat for reparametrised error
rpen = pen.copy()
rpen = rpen[rpen[:,0] >= eps]
rpen[:,1] = rpen[:,1] / rpen[:,0]

# force monotonicity by copying the previous largest entry
# skip entries close to zero (numerical instability)
rpen_bound = rpen[rpen[:,0] >= eps].copy()
rpen_bound[:,1] = np.maximum.accumulate(rpen_bound[:,1])

# plot
plt.figure()
plt.plot(rpen[:,0], rpen[:,1], ".c", label="raw")
plt.plot(rpen_bound[:,0], rpen_bound[:,1], "k", label="max")
plt.xlabel("Input Norm ||x||")
plt.ylabel("Input-Scaled Error ||f(x)-f'(x)|| / ||x||")
plt.legend(loc="upper left")

# compute empirical lipschitz constant of dataset
print("Computing empirical Lipschitz constant")
emp_lips = empirical_lipschitz(device, loader, p_norm=2)

# max empirical constant found
max_emp_lip = np.max(emp_lips)
print("Max constant found:", max_emp_lip)



