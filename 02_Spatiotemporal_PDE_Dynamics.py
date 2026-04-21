import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D

class GutReactorPhysics:
    def __init__(self):
        self.L, self.N = 1.0, 50
        self.dx = self.L / self.N
        self.D_B, self.D_P, self.v = 1e-3, 1e-2, 0.5
        self.mu_B, self.alpha, self.mu_E, self.K_kill = 0.8, 0.5994, 0.5, 2.0

    def pde_system(self, t, y):
        B, P, E = y[0:self.N], y[self.N:2*self.N], y[2*self.N:3*self.N]
        dBdt, dPdt, dEdt = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)

        for i in range(1, self.N-1):
            diff_B = self.D_B * (B[i+1] - 2*B[i] + B[i-1]) / (self.dx**2)
            diff_P = self.D_P * (P[i+1] - 2*P[i] + P[i-1]) / (self.dx**2)
            adv_B = -self.v * (B[i] - B[i-1]) / self.dx
            adv_P = -self.v * (P[i] - P[i-1]) / self.dx
            inhibition = (P[i]**2) / (P[i]**2 + self.K_kill**2)

            dBdt[i] = diff_B + adv_B + (self.mu_B * B[i] * (1 - B[i]/100.0))
            dPdt[i] = diff_P + adv_P + (self.alpha * B[i] - 0.3 * P[i])
            dEdt[i] = (self.mu_E * E[i] * (1 - E[i]/100.0)) - (1.5 * inhibition * E[i])

        return np.concatenate([dBdt, dPdt, dEdt])

physics = GutReactorPhysics()
y0_B, y0_P, y0_E = np.zeros(physics.N), np.zeros(physics.N), np.ones(physics.N) * 80.0
y0_B[0:5] = 50.0
y0 = np.concatenate([y0_B, y0_P, y0_E])

t_eval = np.linspace(0, 10, 200)
solution = solve_ivp(physics.pde_system, (0, 10), y0, t_eval=t_eval, method='LSODA')

E_res = solution.y[2*physics.N:3*physics.N, :]

resistome_entropy = [entropy(E_res[:, t] / (np.sum(E_res[:, t]) + 1e-9)) for t in range(len(t_eval))]
energy = np.sum(E_res, axis=0)
lyapunov_series = np.log((energy[1:] + 1e-9) / (energy[:-1] + 1e-9))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(t_eval, np.linspace(0, 1, physics.N))
surf = ax.plot_surface(X, Y, E_res, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('Time (Months)')
ax.set_ylabel('Gut Location')
ax.set_zlabel('E. coli Abundance')
plt.show()
