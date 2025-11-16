# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:05:10 2024

@author: Edoardo
"""

import numpy as np

# load the dataset with mu = 1.0
data = np.loadtxt("./van_der_pol_2.csv", delimiter=",", skiprows=1).astype(np.float32)

# split the simulation into separate trajectories
sim_len = 3001
n_traj = int(data.shape[0] / sim_len)
traj = np.reshape(data, [n_traj, sim_len, 6])

# downsample by discarding the end of each trajectory
trunc_len = int(sim_len / 50)
traj = traj[:, :trunc_len, :]

# compute the finite difference pairs
n_pairs = n_traj * (trunc_len - 1)
t_p0 = traj[:, :-1, -2:].reshape([n_pairs, 2])
t_p1 = traj[:, 1:, -2:].reshape([n_pairs, 2])

# write the preprocessed data
output = np.column_stack([t_p0, t_p1])
np.savetxt("./vdp_clean.csv", output, delimiter=",")
