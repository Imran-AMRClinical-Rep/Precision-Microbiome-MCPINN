import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class TreatmentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, t):
        return self.net(t / 12.0) * 2.0

def stochastic_gut_simulation(policy_net, noise_level=0.05, n_paths=100):
    T, dt = 12.0, 0.1
    steps = int(T / dt)
    time_grid = torch.linspace(0, T, steps).view(-1, 1)

    B_list, P_list, E_list = [torch.full((n_paths, 1), 5.0)], [torch.full((n_paths, 1), 0.0)], [torch.full((n_paths, 1), 80.0)]
    dose_list = []

    for t in range(steps - 1):
        B_curr, P_curr, E_curr = B_list[-1], P_list[-1], E_list[-1]
        dose = policy_net(time_grid[t]).expand(n_paths, 1)
        dose_list.append(dose)

        dB = (0.5 * B_curr * (1 - B_curr/100.0) + 5.0 * dose) * dt
        dP = (0.5994 * B_curr - 0.3 * P_curr) * dt
        inhibit = (P_curr**2) / (P_curr**2 + 4.0)
        dE = (0.5 * E_curr * (1 - E_curr/100.0) - 1.5 * inhibit * E_curr) * dt

        noise_B = torch.randn_like(B_curr) * noise_level * np.sqrt(dt) * B_curr
        noise_E = torch.randn_like(E_curr) * noise_level * np.sqrt(dt) * E_curr

        B_list.append(torch.relu(B_curr + dB + noise_B))
        P_list.append(P_curr + dP)
        E_list.append(torch.relu(E_curr + dE + noise_E))

    return time_grid, torch.cat(B_list, dim=1), torch.cat(P_list, dim=1), torch.cat(E_list, dim=1), torch.cat(dose_list, dim=1)

policy = TreatmentPolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.005)

for epoch in range(500):
    optimizer.zero_grad()
    _, _, _, E, doses = stochastic_gut_simulation(policy, noise_level=0.05, n_paths=50)
    loss = torch.mean(E**2) + torch.mean(E[:, -1]**2) * 5.0 + torch.mean(doses**2) * 0.1
    loss.backward()
    optimizer.step()

with torch.no_grad():
    t_final, _, _, E_final, D_final = stochastic_gut_simulation(policy, noise_level=0.05, n_paths=100)

t_np = t_final.numpy().flatten()
E_np = E_final.numpy()
mean_E, std_E = np.mean(E_np, axis=0), np.std(E_np, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(t_np, mean_E, 'r-', linewidth=3)
plt.fill_between(t_np, mean_E - 2*std_E, mean_E + 2*std_E, color='red', alpha=0.2)
plt.axhline(5.0, color='black', linestyle='--')
plt.show()
