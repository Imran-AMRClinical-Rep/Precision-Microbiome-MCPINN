import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

torch.manual_seed(42)
np.random.seed(42)

def true_biological_system(y, t):
    B, P, E = y
    mu_B, alpha, decay, mu_E = 0.8, 0.6, 0.3, 0.5
    inhibition = (P**2) / (P**2 + 4.0)
    dBdt = mu_B * B * (1 - B/100.0)
    dPdt = (alpha * B) - (decay * P)
    dEdt = mu_E * E * (1 - E/100.0) - (1.5 * inhibition * E)
    return [dBdt, dPdt, dEdt]

t_true = np.linspace(0, 12, 200)
y0 = [5.0, 0.0, 50.0]
solution = odeint(true_biological_system, y0, t_true)

sample_indices = [10, 50, 100, 150, 199]
t_clinical = torch.tensor(t_true[sample_indices], dtype=torch.float32).view(-1, 1)
y_clinical = torch.tensor(solution[sample_indices], dtype=torch.float32)

class MC_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
        self.alpha_hat = nn.Parameter(torch.tensor([0.1]))
        self.kill_hat = nn.Parameter(torch.tensor([0.1]))

    def forward(self, t):
        return self.net(t)

def calculate_loss(model, t_collocation, t_clinical, y_clinical):
    y_pred_clinical = model(t_clinical)
    loss_data = torch.mean((y_pred_clinical - y_clinical)**2)

    t_collocation.requires_grad = True
    y_pred = model(t_collocation)
    B, P, E = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    grad_outputs = torch.ones_like(B)
    dBdt = torch.autograd.grad(B, t_collocation, grad_outputs, create_graph=True)[0].flatten()
    dPdt = torch.autograd.grad(P, t_collocation, grad_outputs, create_graph=True)[0].flatten()
    dEdt = torch.autograd.grad(E, t_collocation, grad_outputs, create_graph=True)[0].flatten()

    f_B = 0.8 * B * (1 - B/100.0)
    f_P = (model.alpha_hat * B) - (0.3 * P)
    f_E = 0.5 * E * (1 - E/100.0) - (model.kill_hat * P * E)

    loss_ode = torch.mean((dBdt - f_B)**2) + torch.mean((dPdt - f_P)**2) + torch.mean((dEdt - f_E)**2)
    loss_mass = torch.mean(torch.relu(dPdt - (model.alpha_hat * B))**2)

    return loss_data + loss_ode + (10.0 * loss_mass)

model = MC_PINN()
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.005)
t_collocation = torch.linspace(0, 12, 100).view(-1, 1)

for epoch in range(5000):
    optimizer.zero_grad()
    loss = calculate_loss(model, t_collocation, t_clinical, y_clinical)
    loss.backward()
    optimizer.step()

alpha_val = model.alpha_hat.detach().item()
kill_val = model.kill_hat.detach().item()

def c_section_simulation(y, t, learned_alpha, learned_kill, intervention=False):
    B, P, E = y
    mu_B = 1.2 if intervention else 0.3
    inhibition = (P**2) / (P**2 + 4.0)
    dBdt = mu_B * B * (1 - B/100.0)
    dPdt = (learned_alpha * B) - (0.3 * P)
    dEdt = 0.5 * E * (1 - E/100.0) - (1.5 * inhibition * E)
    return [dBdt, dPdt, dEdt]

t_sim = np.linspace(0, 12, 100)
y0_c_section = [0.1, 0.0, 90.0]
sol_untreated = odeint(c_section_simulation, y0_c_section, t_sim, args=(alpha_val, kill_val, False))
sol_treated = odeint(c_section_simulation, y0_c_section, t_sim, args=(alpha_val, kill_val, True))

plt.figure(figsize=(10, 6))
plt.plot(t_sim, sol_untreated[:, 2], 'r--', linewidth=2, label="C-Section (Untreated)")
plt.plot(t_sim, sol_treated[:, 2], 'g-', linewidth=3, label="C-Section + Probiotic (Treated)")
plt.axhline(y=5.0, color='k', linestyle=':', label="Safety Threshold")
plt.xlabel("Time (Months)")
plt.ylabel("E. coli (ARG) Abundance")
plt.legend()
plt.show()
