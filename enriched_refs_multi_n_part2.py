'''
In this file, we run the 10 trials per sample of quick_vis_n_regs.ipynb with the same training data, but this time 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import time
from pathlib import Path
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print("Running on:", device)

# Reinitialise hyperparameters
input_dim, width, output_dim = 4, 500, 1
start_rate, final_rate, num_epochs = 0.1, 0.000000001, 40000
beta = 100
sigma = beta * math.sqrt(0.1)
p = 2
ns = [16, 32, 64]
num_sims = 10
T = 50
L_x = 20
theta = 0.1
mu = 0
wind_sigma = 0.05
tau = torch.tensor(1, device = device)
A = 2 # controls how soft/hard the obstacle cost is
M = 10 # controls how high the cost is
vs = 0.8 # speed of boat

def running_cost(x, y, A, M):
    return( 1 - 1/(1 + torch.exp(A * (1 - x**2 - y**2)))) * M

def terminal_cost(x, y, L_x):
    return torch.norm(x - L_x, dim = 1, keepdim = True)**2 + torch.norm(y, dim = 1, keepdim = True)**2

def regularising(th1, th2, p, r):
    full_th = torch.cat((th1, th2))
    nrm = torch.norm(full_th)
    return nrm**p + 1e-9 * torch.exp(nrm)

def regularising_t(theta1, theta2, p, beta):
    theta_vmap = torch.vmap(regularising, in_dims=(0, 1, None, None))
    ells_by_j = theta_vmap(theta1, theta2, p, theta1.shape[0])
    return torch.sum(ells_by_j) / (2 * beta**2)

def gen_ref_path(reference_control, p0, winds, vs, T, n):
    ref_path = torch.zeros(n, T+1, 2, device = device) # use a 3D tensor to store path information
    ref_path[:, 0, :] = p0 
    for t in range(T): 
        heading = torch.cat([torch.cos(reference_control), torch.sin(reference_control)], dim = 1)
        wind_vec = torch.cat([torch.zeros(n, device = device).view(n, 1), winds[:, t].view(n, 1)], dim = 1)
        ref_path[:, t+1, :] = ref_path[:, t] + vs * heading + wind_vec
    return ref_path

def gen_ref_path_pointing(p0, pT, winds, vs, T, n):
    pointing_path = torch.zeros(n, T+1, 2, device = device)
    pointing_path[:, 0, :] = p0
    pointing_ctrls = torch.zeros(n, T, 1, device = device)
    for t in range(T):
        heading = torch.cat([torch.cos(pointing_ctrls[:, t, :]), torch.sin(pointing_ctrls[:, t, :])], dim = 1)
        wind_vec = torch.cat([torch.zeros(n, device = device).view(n, 1), winds[:, t].view(n, 1)], dim = 1)
        pointing_path[:, t+1, :] = pointing_path[:, t] + vs * heading + wind_vec
        new_dir = (pT - pointing_path[:, t+1, :])
        angle = torch.atan2(new_dir[:, 1], new_dir[:, 0])
        if t < T-1:
            pointing_ctrls[:, t+1, 0] = angle
    return pointing_path, pointing_ctrls

class NeuralNet_entropy(nn.Module):
        def __init__(self, input_dim, width, output_dim):
            super(NeuralNet_entropy, self).__init__()
            self.hidden_layer = nn.Linear(input_dim, width)
            self.hidden_layer.bias.data.zero_()
            self.hidden_layer.bias.requires_grad = False
            self.sigmoid = nn.Tanh()
            self.output_layer = nn.Linear(width, output_dim)
            self.output_layer.bias.data.zero_()
            self.output_layer.bias.requires_grad = False
            self.width = width
        def forward(self, x):
            activations = self.sigmoid(self.hidden_layer(x))
            unscaled = self.output_layer(activations)
            return unscaled/self.width
        
num_sims = 10
ref_batches = 4
for sim in range(10, 2 * num_sims):
    for n in ns:
        print(f"Simulation {sim+1}/{num_sims} for n = {n}")
        # retrieve training data and gather reference trajectories
        # pattern = f"*_n{n}_sim{sim+1}_of_10.pt"
        # try:
        #     with open(f"/home/baros/GitHub Repos/Zermelo/differing_n_regs/{pattern}", "rb") as f:
        #         data = torch.load(f, map_location = device)
        # except:
        #     with open(f"./differing_n_regs/{pattern}", "rb") as f:
        #         data = torch.load(f, map_location = device)
        pattern = f"*_n{n}_sim{sim+1}_of_10.pt"

        search_dirs = [
            Path("/home/baros/GitHub Repos/Zermelo/differing_n_regs"),
            Path("./differing_n_regs"),
        ]

        data = None

        for directory in search_dirs:
            matches = list(directory.glob(pattern))

            if matches:
                file_path = matches[0]   # take first matching file
                data = torch.load(file_path, map_location=device)
                break

        training_winds = data["training_data"]
        ref_ctrl = torch.zeros(n, 1, device = device)
        initial_points = torch.zeros(n, 2, device = device) - torch.tensor([20, 0], device = device)
        ref_path = gen_ref_path(ref_ctrl, initial_points, training_winds, vs, T, n)
        ref_path = torch.cat([ref_path, 
                            gen_ref_path_pointing(initial_points,
                                                    torch.zeros(n, 2, device = device) + torch.tensor([20, 2.5], device = device),
                                                    training_winds,
                                                    vs,
                                                    T, 
                                                    n)[0]],
                                                    dim = 0)
        ref_path = torch.cat([ref_path, 
                            gen_ref_path_pointing(initial_points,
                                                    torch.zeros(n, 2, device = device) + torch.tensor([20, 0], device = device),
                                                    training_winds,
                                                    vs,
                                                    T, 
                                                    n)[0]],
                                                    dim = 0)
        ref_path = torch.cat([ref_path, 
                            gen_ref_path_pointing(initial_points,
                                                    torch.zeros(n, 2, device = device) + torch.tensor([20, -2.5], device = device),
                                                    training_winds,
                                                    vs,
                                                    T, 
                                                    n)[0]],
                                                    dim = 0)
        training_winds = training_winds.repeat((ref_batches, 1))
        zeros_vec, ones_vec = torch.zeros(n * ref_batches, 1, device = device), torch.ones(n * ref_batches, 1, device = device) # just for later
        # set up models
        models_reg = [NeuralNet_entropy(input_dim, width, output_dim).to(device) for _ in range(T)]
        optimisers = [optim.AdamW(model.parameters(), lr=start_rate) for model in models_reg]
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max = num_epochs, eta_min = final_rate) for opt in optimisers]
        # training
        for t in reversed(range(T)):
            print(f"Backwards Inductive Step t = {t}...")
            path_length = T - t
            for epoch in range(num_epochs):
                final_c = 0
                current_paths = [ref_path[:, t, :]]
                # generate path from T
                for futs in range(path_length):
                    angle = models_reg[t + futs](torch.cat([current_paths[-1]/torch.tensor([20, 10], device = device),
                                                        training_winds[:, t + futs].view(n * ref_batches, 1), ones_vec.view(n * ref_batches, 1)], dim = 1))
                    heading = torch.cat([torch.cos(angle),
                                        torch.sin(angle)], dim = 1)
                    wind_vec = torch.cat([zeros_vec,
                                        training_winds[:, t + futs].view(n * ref_batches, 1)], dim = 1)
                    new_p = current_paths[-1] + vs * heading + wind_vec
                    current_paths.append(new_p)
                # stack current_paths into a 3D tensor
                current_paths = torch.stack(current_paths, dim = 1)
                # compute losses
                # for running cost, exclude initial point (uncontrolled)
                running_c = running_cost(current_paths[:, 1:, 0], current_paths[:, 1:, 1], A, M)
                running_c = torch.sum(running_c, dim = 1)
                terminal_c = terminal_cost(current_paths[:, -1, 0].view(n * ref_batches, 1), current_paths[:, -1, 1].view(n * ref_batches,1), L_x).view(n * ref_batches, 1)
                reg = regularising_t(
                    models_reg[t].hidden_layer.weight,
                    models_reg[t].output_layer.weight,
                    p,
                    beta
                )
                final_c = torch.mean(running_c + terminal_c) * width + reg 
                # backprop and update
                final_c.backward()
                optimisers[t].step()
                
                # inject noise
                with torch.no_grad():
                    lr_t = optimisers[t].param_groups[0]["lr"]
                    noise_scale = math.sqrt(lr_t) * sigma / beta
                    models_reg[t].hidden_layer.weight.data += (
                        noise_scale * torch.randn_like(models_reg[t].hidden_layer.weight)
                    )
                    models_reg[t].output_layer.weight.data += (
                        noise_scale * torch.randn_like(models_reg[t].output_layer.weight)
                    )
                # reset gradient and step scheduler
                optimisers[t].zero_grad()
                schedulers[t].step()
                if epoch % 1000 == 0:
                    with torch.no_grad():
                        print(f" Epoch: {epoch}, Obstacle Cost: {running_c.mean().item():.6f}, Terminal Cost: {terminal_c.mean().item():.6f}")
            # freeze model after training
            for param in models_reg[t].parameters():
                param.requires_grad = False
        # write to dictionary and save for each n and simulation
        entropy_info = {
            "date": time.strftime("%Y-%m-%d", time.localtime()),
            "n": n,
            "T": T,
            "L_x": L_x, 
            "theta": theta,
            "mu": mu,
            "wind_sigma": wind_sigma,
            "tau": tau,
            "A": A,
            "M": M,
            "beta": beta,
            "sigma": sigma,
            "activation_function": "tanh",
            "network_width": width,
            "start_rate": start_rate,
            "final_rate": final_rate,
            "num_epochs": num_epochs,
            "optimiser": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "vs": vs,
            "training_data": training_winds,
            "entropy_model_parameters": [model.state_dict() for model in models_reg],
            "method": "Entropy-Regularised"
        }
        # save info, creating folder if it doesn't exist
        save_dir = "differing_n_regs_refs_enriched"
        os.makedirs(save_dir, exist_ok=True)
        date_str = time.strftime('%Y-%m-%d', time.localtime())
        fname = os.path.join(save_dir, f"{date_str}_n{n}_sim{sim+1}_of_{num_sims}.pt")
        torch.save(entropy_info, fname)