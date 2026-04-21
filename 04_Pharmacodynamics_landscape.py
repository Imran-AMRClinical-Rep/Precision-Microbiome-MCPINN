import numpy as np
import matplotlib.pyplot as plt

def efflux_kinetics(E_coli_load, ALA_conc, pump_expression=100.0):
    V_max = pump_expression
    Km_toxin, Ki_ALA, Influx = 50.0, 10.0, 100.0
    inhibition_factor = 1 + (ALA_conc / Ki_ALA)
    Efflux_Activity = (V_max * 100.0) / (Km_toxin * inhibition_factor + 100.0)
    return Influx / (Efflux_Activity + 1.0)

ala_levels = np.linspace(0, 100, 100)
arg_levels = np.linspace(0, 200, 100)
X, Y = np.meshgrid(ala_levels, arg_levels)
Z = np.zeros_like(X)

for i in range(100):
    for j in range(100):
        toxin = efflux_kinetics(1.0, X[i,j], Y[i,j])
        Z[i,j] = 1.0 / (1.0 + np.exp(0.5 * (toxin - 15.0)))

plt.figure(figsize=(11, 8))
contour = plt.contourf(X, Y, Z, 20, cmap='RdYlBu_r')
plt.colorbar(contour, label='E. coli Survival Probability')
plt.xlabel('[ALA] Postbiotic Concentration (mM)')
plt.ylabel('Efflux Pump (ARG) Expression Level')
plt.show()
