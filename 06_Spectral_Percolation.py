import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

def simulate_network_collapse(n_species=200, initial_connectivity=0.05):
    G = nx.erdos_renyi_graph(n_species, initial_connectivity, seed=42)
    dose_range = np.linspace(0, 1.0, 50)
    network_integrity, spectral_radius = [], []

    for dose in dose_range:
        G_temp = G.copy()
        edges = list(G_temp.edges())
        num_to_remove = int(len(edges) * dose)
        
        np.random.seed(42)
        indices = np.random.choice(len(edges), num_to_remove, replace=False)
        for i in indices:
            G_temp.remove_edge(*edges[i])

        if len(G_temp.edges()) > 0:
            components = [len(c) for c in nx.connected_components(G_temp)]
            S = max(components) / n_species
            adjacency = nx.adjacency_matrix(G_temp).todense()
            eigenvalues = np.linalg.eigvals(adjacency)
            rho = max(abs(eigenvalues))
        else:
            S, rho = 0.0, 0.0
            
        network_integrity.append(S)
        spectral_radius.append(rho)

    return dose_range, network_integrity, spectral_radius

doses, integrity, radius = simulate_network_collapse()

fig, ax1 = plt.subplots(figsize=(12, 8))
color_1, color_2 = '#1f77b4', '#d62728'

ax1.set_xlabel('ALA Intervention Intensity', fontsize=14, fontweight='bold')
ax1.set_ylabel('Size of Giant Resistance Cluster (S)', color=color_1, fontsize=14, fontweight='bold')
ax1.plot(doses, integrity, color=color_1, linewidth=5)
ax1.fill_between(doses, integrity, color=color_1, alpha=0.1)

ax2 = ax1.twinx()
ax2.set_ylabel('Spectral Radius (Spread Velocity)', color=color_2, fontsize=14, fontweight='bold')
ax2.plot(doses, radius, color=color_2, linestyle='--', linewidth=3)

critical_indices = np.where(np.array(integrity) < 0.05)[0]
critical_dose = doses[critical_indices[0]] if len(critical_indices) > 0 else 0.94

plt.axvline(critical_dose, color='black', linestyle=':', linewidth=3)
plt.text(critical_dose - 0.05, max(radius)*0.8, f"CRITICAL\nPERCOLATION\nTHRESHOLD\n(Pc = {critical_dose:.2f})",
         ha='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.show()
