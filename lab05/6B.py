import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load C-V data
# =========================
# Change this path if needed
file_path = "lab05\ece350_FDSOI_MOSCAP_CV_Characteristics.csv"

df = pd.read_csv(file_path)

# Print columns once so you can confirm names
print(df.columns)

# =========================
# Set the correct column names here
# =========================
# Replace these with the actual names printed above if needed
vg = df.iloc[:, 0].astype(float).values
cv = df.iloc[:, 1].astype(float).values * 1e15  # convert to fF

# remove NaN just in case
mask = ~np.isnan(cv)
vg = vg[mask]
cv = cv[mask]

c_min = np.min(cv)
c_max = np.max(cv)

plt.figure(figsize=(8,5))

plt.plot(vg, cv, 'o-', color='red', label='Measured HF C-V')

plt.axhline(c_max, color='blue', linestyle='--',
            label=f'C_max = {c_max:.2f} fF')

plt.axhline(c_min, color='green', linestyle='--',
            label=f'C_min = {c_min:.2f} fF')

plt.title('HF C-V Characteristics of n-FDSOI')
plt.xlabel('Gate Voltage VG (V)')
plt.ylabel('Capacitance $C_{var}$ (fF)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()