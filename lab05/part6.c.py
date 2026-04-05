import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ece350_FinFET_MOSCAP_CV_Characteristics.csv')
vg = df['Vg [V]'].values
cvar = df['Cvar [F]'].values

# Extract Max and Min Capacitance
c_max = np.max(cvar)
c_min = np.min(cvar)

# Device Parameters
n_fins = 512
l_g = 11e-9          
w_eff_per_fin = 95e-9   
w_eff_total = n_fins * w_eff_per_fin
area_total = l_g * w_eff_total

# Constants
epsilon_0 = 8.854e-12   
epsilon_ox = 3.9         
epsilon_si = 11.7    

# Parameter Extractions
# t_OXE
t_oxe = (epsilon_ox * epsilon_0 * area_total) / c_max

# 1/C_min = 1/C_max + 1/C_dep_si
c_dep_si = (c_min * c_max) / (c_max - c_min)

# t_fin
t_fin = (epsilon_si * epsilon_0 * area_total) / c_dep_si

# Plot
plt.figure(figsize=(10, 6))
plt.plot(vg, cvar * 1e15, 'ro-', linewidth=2, label='Measured C-V')
plt.axhline(c_max * 1e15, color='blue', linestyle='--', label=f'C_max = {c_max*1e15:.2f} fF')
plt.axhline(c_min * 1e15, color='green', linestyle='--', label=f'C_min = {c_min*1e15:.2f} fF')

plt.title('HF C-V Characteristics of n-FinFET', fontsize=14)
plt.xlabel('Gate Voltage $V_G$ (V)', fontsize=12)
plt.ylabel('Capacitance $C_{var}$ (fF)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('FinFET_CV_Analysis.png')

# Output results
print(f"FinFET C-V Analysis Results")
print(f"Max Capacitance (C_max): {c_max*1e15:.2f} fF")
print(f"Min Capacitance (C_min): {c_min*1e15:.2f} fF")
print(f"Total Gate Area: {area_total:.4e} m^2")
print(f"Extracted t_OXE: {t_oxe*1e9:.4f} nm")
print(f"Extracted t_fin: {t_fin*1e9:.4f} nm")
