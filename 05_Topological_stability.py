import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_derivatives(B, E, dose, alpha=0.5994):
    P = (alpha * B) / 0.3
    inhibition = (P**2) / (P**2 + 4.0)
    dB = 0.5 * B * (1 - B/100.0) + 5.0 * dose
    dE = 0.5 * E * (1 - E/100.0) - 1.5 * inhibition * E
    return dB, dE

dose_range = np.linspace(0, 0.5, 200)
final_states = []

for dose in dose_range:
    B, E = 5.0, 80.0
    dt = 0.1
    for _ in range(500):
        dB, dE = get_derivatives(B, E, dose)
        B, E = max(0, B + dB * dt), max(0, E + dE * dt)
    final_states.append(E)

plt.figure(figsize=(10, 6))
plt.scatter(dose_range, final_states, c=final_states, cmap='RdYlGn_r', s=10, alpha=0.6)
plt.xlabel('Prebiotic Dosage Input')
plt.ylabel('Final Pathogen Load Equilibrium')
plt.show()

def calculate_potential(x, y):
    V_healthy = -np.exp(-((x - 2)**2 + (y + 1)**2))
    V_sick = -np.exp(-((x + 1)**2 + (y - 2)**2))
    return V_healthy + V_sick + 0.05 * (x**2 + y**2)

x, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = calculate_potential(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.magma, alpha=0.9)
ax.set_xlabel('Bifidobacterium Fitness')
ax.set_ylabel('Pathogen Resistance')
ax.set_zlabel('Quasi-Potential Energy (U)')
plt.show()
