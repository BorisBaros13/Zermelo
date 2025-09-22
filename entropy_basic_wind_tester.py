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

n = 64 # sample size
T = 50 # terminal time
L_x = 20 # target x
theta = 0.1 # mean-reversion
mu = 0 # mean
wind_sigma = 0.05 # volatility for synthesising wind
tau = torch.tensor(1, device = device) # time step
vs = 0.8 # boat speed

A = 2 # controls how soft/hard the obstacle cost is
M = 10 # controls how high the cost is

# cost functions
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

# reference control and initial points
def gen_ref_path(reference_control, p0, winds, vs, T, n):
    ref_path = torch.zeros(n, T+1, 2, device = device) # use a 3D tensor to store path information
    ref_path[:, 0, :] = p0 
    for t in range(T): 
        heading = torch.cat([torch.cos(reference_control), torch.sin(reference_control)], dim = 1)
        wind_vec = torch.cat([torch.zeros(n, device = device).view(n, 1), winds[:, t].view(n, 1)], dim = 1)
        ref_path[:, t+1, :] = ref_path[:, t] + vs * heading + wind_vec
    return ref_path

# model
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
    
# now to train
# model (and regularisation) hyperparameters
input_dim, width, output_dim = 3, 100, 1
start_rate, final_rate, num_epochs = 1, 0.000000001, 20000
beta = 100
sigma = beta * math.sqrt(0.1)
p = 2
models = [NeuralNet(input_dim, width, output_dim).to(device) for _ in range(T)]
optimisers = [optim.AdamW(model.parameters(), lr=start_rate) for model in models]
schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max = num_epochs, eta_min = final_rate) for opt in optimisers]

ref_ctrl = torch.zeros(n, 1, device = device)
initial_points = torch.zeros(n, 2, device = device) - torch.tensor([20, 0], device = device)
training_winds = wind_process(T, theta, mu, wind_sigma, n, tau)
ref_path = gen_ref_path(ref_ctrl, initial_points, training_winds, vs, T, n)
# zeros and ones vectors so we don't keep having to remake it
zeros_vec, ones_vec = torch.zeros(n, 1, device = device), torch.ones(n, 1, device = device)
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
        reg = regularising_t(
            models[t].hidden_layer.weight,
            models[t].output_layer.weight,
            p,
            beta
        )
        final_c = torch.mean(running_c + terminal_c) * width + reg 
        # backprop and update
        final_c.backward()
        optimisers[t].step()
        
        # inject noise
        lr_t = optimisers[t].param_groups[0]["lr"]
        noise_scale = math.sqrt(lr_t) * sigma / beta
        models[t].hidden_layer.weight.data += (
            noise_scale * torch.randn_like(models[t].hidden_layer.weight)
        )
        models[t].output_layer.weight.data += (
            noise_scale * torch.randn_like(models[t].output_layer.weight)
        )
        # reset gradient and step scheduler
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

# gif will go here

# in and out-of-sample plots
test_size = 1000
test_winds = wind_process(T, theta, mu, wind_sigma, test_size, tau)
test_initial_points = torch.zeros(test_size, 2, device = device) - torch.tensor([20, 0], device = device)
test_paths = [test_initial_points]
zeros_vec = torch.zeros(test_size, 1, device = device)
# forward rollout
for t in range(T):
    wind_vec = torch.cat([zeros_vec,
                          test_winds[:, t].view(test_size, 1)], dim = 1)
    angle = models[t](torch.cat([test_paths[-1]/torch.tensor([20, 10], device = device),
                                test_winds[:, t].view(test_size, 1)], dim = 1))
    heading = torch.cat([torch.cos(angle), torch.sin(angle)], dim = 1)
    new_p = test_paths[-1] + vs * heading + wind_vec
    test_paths.append(new_p)
test_paths = torch.stack(test_paths, dim = 1)
# plotting 
red = mpatches.Patch(color="red", label="Out-of-Sample")
green = mpatches.Patch(color="green", label="In-Sample")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=800,
                       constrained_layout = True)
ax.set_ylim([-6, 6])
ax.set_xlim(-L_x, L_x)
# add the obstacle contour 
grid_res = 300  # Resolution of the grid
x_vals = torch.linspace(-6, 6, grid_res)
y_vals = torch.linspace(-6, 6, grid_res)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')
Z = running_cost(X, Y, A=2, M=10)*100
Z[Z < 1.5] = float('nan')
ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=10, cmap='Greys', alpha=0.8)


for pth in range(test_size):
    ax.plot(test_paths[pth, :, 0].cpu().detach(),
             test_paths[pth, :, 1].cpu().detach(), 
             linewidth = 0.9, alpha = 0.5, color = "red")
for pth in range(n):
    ax.plot(current_paths[pth, :, 0].cpu().detach(), current_paths[pth, :, 1].cpu().detach(), 
            linewidth = 0.9, color = "green")
targ = ax.scatter(x=20, y=0, label="Target", color="black", zorder=1000)
ax.set_title(f"Regularised Problem, Beta = {beta:.3f}, Sigma = {sigma:.3f}")
ax.legend(handles=[red, green, targ])
plt.savefig("entropy_in_out_sample.png", dpi=600, bbox_inches='tight')
plt.close()

# histograms
train_loss = np.log(np.array(terminal_cost(current_paths[:, -1, 0].view(n, 1),
                           current_paths[:, -1, 1].view(n, 1), L_x).detach().cpu())[:, 0])
test_loss = np.log(np.array(terminal_cost(test_paths[:, -1, 0].view(test_size, 1),
                           test_paths[:, -1, 1].view(test_size, 1), L_x).detach().cpu())[:, 0])
fig, ax = plt.subplots(figsize = (19.2, 10.8), constrained_layout = True)
ax.hist(test_loss, bins = 20, density = True, alpha = 0.6, color = "blue", label = "Out-of-Sample")
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.8)
ax.hist(train_loss, bins = 20, density = True, alpha = 0.6, color = "orange", label = "In-Sample")
tick_location_1 = np.mean(test_loss)
ax.axvline(x=tick_location_1, color='blue', linewidth=2, linestyle = "-")
tick_location_2 = np.mean(train_loss)
ax.axvline(x=tick_location_2, color='orange', linewidth=2)
ax.set_xlabel("Log of Squared Distance")
ax.set_title(f"Histogram of (Log of) Terminal Costs for Regularised Problem, Beta = {beta:.3f}, Sigma = {sigma:.3f}")
ax.legend()

plt.savefig("entropy_hists_sample.png", dpi=300, bbox_inches='tight')
plt.close()