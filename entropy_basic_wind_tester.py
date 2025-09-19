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

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print("Running on:", device)

def wind_process(T, theta, mu, wind_sigma, n, tau):
    num_to_sim = int(T / tau)
    winds = torch.zeros(n, num_to_sim + 1, device = device)
    winds[:, 0] = 0.5 * torch.rand(n , device = device) - 0.25 
    for step in range(1, num_to_sim + 1):
        dW = torch.randn(n , device = device)
        winds[:, step] = winds[:, step - 1] +(mu - winds[:, step - 1]) * theta * tau + wind_sigma * torch.sqrt(tau) * dW
    # only return winds for the integer time-steps
    final_wind = winds[:, 1:][:, ::int(1 / tau)]
    # add initial wind to the front
    winds = torch.cat((winds[:, 0].view(n, 1), final_wind), dim = 1)
    return winds

n = 64
T = 50
L_x = 20
theta = 0.1
mu = 0
wind_sigma = 0.05
tau = torch.tensor(1, device = device)

A = 2 # controls how soft/hard the obstacle cost is
M = 10 # controls how high the cost is
def running_cost(x, y, A, M):
    return( 1 - 1/(1 + torch.exp(A * (1 - x**2 - y**2)))) * M

def terminal_cost(x, y, L_x):
    return torch.norm(x - L_x, dim = 1, keepdim = True)**2 + torch.norm(y, dim = 1, keepdim = True)**2

ref_ctrl = torch.zeros(n, 1, device = device)
initial_points = torch.zeros(n, 2, device = device) - torch.tensor([20, 0], device = device)
training_winds = wind_process(T, theta, mu, wind_sigma, n, tau)
def gen_ref_path(reference_control, p0, winds, vs, T, n):
    ref_path = torch.zeros(n, T+1, 2, device = device) # use a 3D tensor to store path information
    ref_path[:, 0, :] = p0 
    for t in range(T): 
        heading = torch.cat([torch.cos(reference_control), torch.sin(reference_control)], dim = 1)
        wind_vec = torch.cat([torch.zeros(n, device = device).view(n, 1), winds[:, t].view(n, 1)], dim = 1)
        ref_path[:, t+1, :] = ref_path[:, t] + vs * heading + wind_vec
    return ref_path

class NeuralNet(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(NeuralNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, width)
        self.sigmoid = nn.Tanh()
        self.output_layer = nn.Linear(width, output_dim)
        
    def forward(self, x):
        activations = self.sigmoid(self.hidden_layer(x))
        unscaled = self.output_layer(activations)
        return unscaled/self.width
    
# now let's train
input_dim, width, output_dim = 3, 100, 1
start_rate, final_rate, num_epochs = 0.00001, 0.000000001, 20000
models = [NeuralNet(input_dim, width, output_dim).to(device) for _ in range(T)]
optimisers = [optim.AdamW(model.parameters(), lr=start_rate) for model in models]
schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max = num_epochs, eta_min = final_rate) for opt in optimisers]

vs = 0.8
n = 64
A = 2
M = 10
ref_ctrl = torch.zeros(n, 1, device = device)
initial_points = torch.zeros(n, 2, device = device) - torch.tensor([20, 0], device = device)
training_winds = wind_process(T, theta, mu, wind_sigma, n, tau)
ref_path = gen_ref_path(ref_ctrl, initial_points, training_winds, vs, T, n)
# zeros and ones vectors so we don't keep having to remake it
zeros_vec, ones_vec = torch.zeros(n, 1, device = device), torch.ones(n, 1, device = device)
# run without training first, just to check correctness of computation
start_time = time.time()
for t in reversed(range(T)):
    print(f"Backwards Inductive Step t = {t}...")
    path_length = T - t
    for epoch in range(num_epochs):
        final_c = 0
        current_paths = [ref_path[:, t, :]]
        # generate path from T
        for futs in range(path_length):
            angle = models[t + futs](torch.cat([current_paths[-1]/torch.tensor([20, 10], device = device),
                                                   training_winds[:, t + futs].view(n, 1)], dim = 1))
            heading = torch.cat([torch.cos(angle),
                                  torch.sin(angle)], dim = 1)
            wind_vec = torch.cat([zeros_vec,
                                   training_winds[:, t + futs].view(n, 1)], dim = 1)
            new_p = current_paths[-1] + vs * heading + wind_vec
            current_paths.append(new_p)
        # stack current_paths into a 3D tensor
        current_paths = torch.stack(current_paths, dim = 1)
        # compute losses
        # for running cost, exclude initial point (uncontrolled)
        running_c = running_cost(current_paths[:, 1:, 0], current_paths[:, 1:, 1], A, M)
        running_c = torch.sum(running_c, dim = 1)
        terminal_c = terminal_cost(current_paths[:, -1, 0].view(n, 1), current_paths[:, -1, 1].view(n ,1), L_x).view(n, 1)
        final_c = torch.mean(running_c + terminal_c)
        # backprop and update
        final_c.backward()
        optimisers[t].step()
        optimisers[t].zero_grad()
        schedulers[t].step()

        if epoch % 1000 == 0:
            with torch.no_grad():
                print(f" Epoch: {epoch}, Obstacle Cost: {running_c.mean().item():.6f}, Terminal Cost: {terminal_c.mean().item():.6f}")
    # freeze model after training
    for param in models[t].parameters():
        param.requires_grad = False
end_time = time.time()
print(f"Total training time: {end_time - start_time:.3f} seconds")
