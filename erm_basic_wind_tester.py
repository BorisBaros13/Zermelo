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

n = 50
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

class NeuralNet_2(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(NeuralNet_2, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, width)
        self.hidden_layer_2 = nn.Linear(width, width)
        self.hidden_layer_3 = nn.Linear(width, width)
        self.hidden_layer_4 = nn.Linear(width, width)
        self.sigmoid = nn.Tanh()
        self.output_layer = nn.Linear(width, output_dim)
        
    def forward(self, x):
        activations_1 = self.sigmoid(self.hidden_layer(x))
        activations_2 = self.sigmoid(self.hidden_layer_2(activations_1))
        activations_3 = self.sigmoid(self.hidden_layer_3(activations_2))
        activations_4 = self.sigmoid(self.hidden_layer_4(activations_3))
        unscaled = self.output_layer(activations_3)
        return unscaled# / self.width
    
# now let's train
input_dim, width, output_dim = 3, 100, 1
start_rate, final_rate, num_epochs = 0.00001, 0.000000001, 20000
models_2 = [NeuralNet_2(input_dim, width, output_dim).to(device) for _ in range(T)]
optimizers_2 = [optim.AdamW(model.parameters(), lr=start_rate) for model in models_2]
schedulers_2 = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max = num_epochs, eta_min = final_rate) for opt in optimizers_2]

vs = 0.8
n = 64
A = 2
M = 10
ref_ctrl = torch.zeros(n, 1, device = device)
initial_points = torch.zeros(n, 2, device = device) - torch.tensor([20, 0], device = device)
training_winds = wind_process(T, theta, mu, wind_sigma, n, tau)
ref_path_2 = gen_ref_path(ref_ctrl, initial_points, training_winds, vs, T, n)
# zeros and ones vectors so we don't keep having to remake it
zeros_vec, ones_vec = torch.zeros(n, 1, device = device), torch.ones(n, 1, device = device)
start_time = time.time()
for t in reversed(range(T)):
    print(f"Backwards Inductive Step t = {t}...")
    path_length = T - t
    for epoch in range(num_epochs):
        final_c = 0
        current_paths_2 = [ref_path_2[:, t, :]]
        # generate path from T
        for futs in range(path_length):
            angle = models_2[t + futs](torch.cat([current_paths_2[-1]/torch.tensor([20, 10], device = device),
                                                   training_winds[:, t + futs].view(n, 1)], dim = 1))
            heading = torch.cat([torch.cos(angle),
                                  torch.sin(angle)], dim = 1)
            wind_vec = torch.cat([zeros_vec,
                                   training_winds[:, t + futs].view(n, 1)], dim = 1)
            new_p = current_paths_2[-1] + vs * heading + wind_vec
            current_paths_2.append(new_p)
        # stack current_paths into a 3D tensor
        current_paths_2 = torch.stack(current_paths_2, dim = 1)
        # compute losses
        # for running cost, exclude initial point (uncontrolled)
        running_c = running_cost(current_paths_2[:, 1:, 0], current_paths_2[:, 1:, 1], A, M)
        running_c = torch.sum(running_c, dim = 1)
        terminal_c = terminal_cost(current_paths_2[:, -1, 0].view(n, 1), current_paths_2[:, -1, 1].view(n ,1), L_x).view(n, 1)
        final_c = torch.mean(running_c + terminal_c)
        # backprop and update
        final_c.backward()
        optimizers_2[t].step()
        optimizers_2[t].zero_grad()
        schedulers_2[t].step()

        if epoch % 1000 == 0:
            with torch.no_grad():
                print(f" Epoch: {epoch}, Obstacle Cost: {running_c.mean().item():.6f}, Terminal Cost: {terminal_c.mean().item():.6f}")
    # freeze model after training
    for param in models_2[t].parameters():
        param.requires_grad = False
end_time = time.time()
print(f"Total training time: {end_time - start_time:.3f} seconds")

# now for the animation
ref_control = torch.zeros(n, 1, device=device)
ref_path = [initial_points.to(device)] 

for pos in range(T):
    wind_y = training_winds[:, pos].view(n, 1).to(device)
    wind_vector = torch.cat([torch.zeros_like(wind_y), wind_y], dim=1)
    heading = torch.cat([torch.cos(ref_control), torch.sin(ref_control)], dim=1)
    velocity = vs * heading +  wind_vector
    new_p = ref_path[-1] + velocity
    ref_path.append(new_p)

# Backward rollout
anim_paths = {}
wind_values = {}

anim_paths[f"{T}"] = ref_path
for t in range(T - 1, -1, -1):
    curr_path = ref_path[:t + 1]
    # curr_wind = [training_data[:, :t + 1]]
    curr_wind = [training_winds[:, i].view(n, 1) for i in range(t + 1)]

    curr_p = curr_path[-1]
    for rem in range(0, T - t):
        wind_val = training_winds[:, t + rem].view(n, 1).to(device)
        wind_vector = torch.cat([torch.zeros_like(wind_val), wind_val], dim=1)
        input_tensor = torch.cat((curr_p/torch.tensor([20, 10], device = device), wind_val), dim=1)

        curr_control = models_2[t + rem](input_tensor)
        heading = torch.cat([torch.cos(curr_control), torch.sin(curr_control)], dim=1)
        velocity = vs * heading + wind_vector
        curr_p = curr_p + velocity

        curr_path.append(curr_p)
        curr_wind.append(wind_val)

    anim_paths[f"{t}"] = curr_path
    wind_values[f"{t}"] = curr_wind

# Prepare data for animation
plt.rcParams['animation.embed_limit'] = 100  # (in MB, e.g., 100 MB)
frame_data = []
for key in sorted(anim_paths.keys(), key=lambda k: int(k)):
    step_list = anim_paths[key]
    wind_list = wind_values.get(key, [])
    positions = torch.stack(step_list, dim=0).detach().cpu()  # (steps, N, 2)
    wind_vals = torch.stack(wind_list, dim=0).squeeze(-1).detach().cpu() if wind_list else None  # (steps, N)
    frame_data.append((int(key), positions, wind_vals))
frame_data.reverse()

# Setup plot
L_x, L_y = 20, 6
num_paths_to_show = n
interval = 100


plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "font.size": 25
})
fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True)
ax.set_xlim(-L_x, L_x)
ax.set_ylim(-L_y, L_y)
ax.plot(20, 0, 'ko', markersize=20, zorder=1000)

# Obstacle contour
grid_res = 300
x_vals = torch.linspace(-6, 6, grid_res)
y_vals = torch.linspace(-6, 6, grid_res)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')
Z = running_cost(X, Y, A=2, M=10) * 100
Z[Z < 1.5] = float('nan')
ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=10, cmap='Greys', alpha=0.8)

# Plot elements
grey_lines = [ax.plot([], [], linestyle='--', color='grey', lw=1)[0] for _ in range(num_paths_to_show)]
collections = []
# Plotting parameters

norm = plt.Normalize(training_winds.min().item(), training_winds.max().item())
cmap = plt.cm.plasma
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Wind Value")

# --- Animation functions ---
def init():
    for line in grey_lines:
        line.set_data([], [])
    for coll in collections:
        coll.remove()
    collections.clear()
    return grey_lines

def update(frame_idx):
    for coll in collections:
        coll.remove()
    collections.clear()

    epoch_idx, data, winds = frame_data[frame_idx]
    ax.set_title(f"Backward Inductive Step = {epoch_idx}")

    for i in range(num_paths_to_show):
        if i >= data.shape[1]:
            continue
        x = data[:, i, 0].numpy()
        y = data[:, i, 1].numpy()

        # Grey past
        grey_lines[i].set_data(x[:epoch_idx + 1], y[:epoch_idx + 1])

        # Color future
        if winds is not None and epoch_idx < data.shape[0] - 1:
            segments = np.stack([x[epoch_idx:], y[epoch_idx:]], axis=-1)
            points = segments[:-1]
            segs = np.stack([points, segments[1:]], axis=1)
            colors = cmap(norm(winds[epoch_idx + 1:, i].numpy()))

            lc = LineCollection(segs, colors=colors, linewidths=2)
            ax.add_collection(lc)
            collections.append(lc)

    return grey_lines + collections

ani = animation.FuncAnimation(
    fig, update, frames=len(frame_data),
    init_func=init, blit=False, interval=interval
)

ani.save("erm_basic_wind_sample_animation.gif", writer="pillow", fps=10)
plt.close()

# now for out-of-sample
test_size = 1000
test_winds = wind_process(T, theta, mu, wind_sigma, test_size, tau)
test_initial_points = torch.zeros(test_size, 2, device = device) - torch.tensor([20, 0], device = device)
test_paths = [test_initial_points]
zeros_vec = torch.zeros(test_size, 1, device = device)
# forward rollout
for t in range(T):
    wind_vec = torch.cat([zeros_vec,
                          test_winds[:, t].view(test_size, 1)], dim = 1)
    angle = models_2[t](torch.cat([test_paths[-1]/torch.tensor([20, 10], device = device),
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
    ax.plot(current_paths_2[pth, :, 0].cpu().detach(), current_paths_2[pth, :, 1].cpu().detach(), 
            linewidth = 0.9, color = "green")
targ = ax.scatter(x=20, y=0, label="Target", color="black", zorder=1000)
ax.set_title("Unregularised Problem")
ax.legend(handles=[red, green, targ])
plt.savefig("erm_in_out_sample.png", dpi=600, bbox_inches='tight')
plt.close()

# now we plot and save the histogram of (terminal) costs too
# first, compute them
train_loss = np.log(np.array(terminal_cost(current_paths_2[:, -1, 0].view(n, 1),
                           current_paths_2[:, -1, 1].view(n, 1), L_x).detach().cpu())[:, 0])
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
ax.set_title("Histogram of (Log of) Terminal Costs for Unregularised Problem")
ax.legend()

plt.savefig("erm_hists_sample.png", dpi=300, bbox_inches='tight')
plt.close()