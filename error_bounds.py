# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:27:45 2024

@author: Edoardo
"""

import numpy as np
import torch

"""
Takes the model and the trainset, returns tuples ||x||_p, ||y-f(x)||_p
"""
def pred_error_norm(model, device, loader, p_norm=2):
    
    norms = np.zeros([len(loader.dataset), 2])
    
    model.eval()
    with torch.no_grad():
    
        i = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            error = model(data) - target
            
            n = data.shape[0]
            norms[i:i+n, 0] = torch.norm(data, p=p_norm, dim=-1).detach().numpy()
            norms[i:i+n, 1] = torch.norm(error, p=p_norm, dim=-1).detach().numpy()
            i = i + n
    
    return norms

# """
# Takes an unsorted list of tuples ||x||, ||err||
# Returns an input-dependent bound ||err|| <= coeff(||x||) * ||x|| + offset
# which is valid for any input x' with ||x'|| <= ||x||
# """
# def cone_coeffs(norms, zero_thresh=1e-4):
    
#     # sort the input in ascending order of ||x||
#     norms = norms[np.argsort(norms[:,0])]
    
#     # the offset is the max error of small-norm inputs
#     small = norms[:, 0] <= zero_thresh
#     offset = np.max(norms[small][:, 1])
    
#     # compute the first guess; it may not be monotonic
#     coeffs = np.maximum(norms[:,1] - offset, 0) / norms[:,0]
    
#     # force monotonicity by copying the previous largest entry
#     coeffs = np.maximum.accumulate(coeffs)
    
#     return np.column_stack((norms[:,0], coeffs)), offset

"""
Takes the trainset, returns tuples ||x||_p, ||y-f(x)||_p
"""
def empirical_lipschitz(device, loader, p_norm=2):
    
    # extract random index pairs (i.j)
    i = np.arange(len(loader.dataset))
    np.random.shuffle(i)
    j = np.concatenate([i[1:], [i[0]]])
    
    data = np.array(loader.dataset)
    x_diff = data[i,0] - data[j,0]
    y_diff = data[i,1] - data[j,1]
    
    x_norm = np.linalg.norm(x_diff, p_norm, axis=-1)
    y_norm = np.linalg.norm(y_diff, p_norm, axis=-1)
    emp_lips = y_norm / (x_norm + 1e-8)
    
    return emp_lips
