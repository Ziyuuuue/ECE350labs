import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------- File --------------------
FILEPATH = "data\One-side\Subt2__0__11-23-12-198.txt"

def read_spectrum(path):
    wl, I = [], []
    start = False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if ">>>>>Begin Spectral Data<<<<<" in line:
                start = True
                continue
            if not start:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                I.append(float(parts[1]))
            except ValueError:
                pass
    wl = np.array(wl)
    I = np.array(I)
    idx = np.argsort(wl)
    return wl[idx], I[idx]

def smooth(y, n=11):
    n = int(n)
    if n < 1:
        return y
    if n % 2 == 0:
        n += 1
    return np.convolve(y, np.ones(n)/n, mode="same")

def fwhm_from_peak_outward(wl, y, peak_idx, window_nm=200):
    """
    Robust FWHM: start at peak and walk outward to find nearest half-max crossings.
    Uses a local baseline (min within window) and linear interpolation at crossings.
    """
    pk_wl = wl[peak_idx]
    mask = (wl >= pk_wl - window_nm/2) & (wl <= pk_wl + window_nm/2)

    # local arrays
    idxs = np.where(mask)[0]
    if len(idxs) < 10:
        return None

    wl_loc = wl[idxs]
    y_loc  = y[idxs]

    # map global peak_idx to local index
    loc_peak = np.where(idxs == peak_idx)[0]
    if len(loc_peak) == 0:
        return None
    loc_peak = int(loc_peak[0])

    baseline = np.min(y_loc)
    pk_val = y[peak_idx]
    half = baseline + 0.5*(pk_val - baseline)

    # --- search left from peak for crossing y drops below half ---
    left_cross = None
    for i in range(loc_peak, 0, -1):
        if y_loc[i] >= half and y_loc[i-1] < half:
            # interpolate between i-1 and i
            x1, y1 = wl_loc[i-1], y_loc[i-1]
            x2, y2 = wl_loc[i],   y_loc[i]
            left_cross = x1 + (half - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            break

    # --- search right from peak for crossing y drops below half ---
    right_cross = None
    for i in range(loc_peak, len(wl_loc)-1):
        if y_loc[i] >= half and y_loc[i+1] < half:
            x1, y1 = wl_loc[i],   y_loc[i]
            x2, y2 = wl_loc[i+1], y_loc[i+1]
            right_cross = x1 + (half - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            break

    if left_cross is None or right_cross is None:
        return None

    return left_cross, right_cross, right_cross-left_cross, pk_wl, pk_val, half

# ---------------- Processing ----------------
wl, I = read_spectrum(FILEPATH)
I_s = smooth(I, n=11)

# Laser peak = global max
laser_idx = int(np.argmax(I_s))
laser_wl = wl[laser_idx]

# PL peak = strongest away from laser
exclude_nm = 25
valid = (wl < laser_wl - exclude_nm) | (wl > laser_wl + exclude_nm)
pl_idx = int(np.argmax(np.where(valid, I_s, -np.inf)))

# FWHM with peak-outward method
laser = fwhm_from_peak_outward(wl, I_s, laser_idx, window_nm=40)   # narrow
pl    = fwhm_from_peak_outward(wl, I_s, pl_idx,    window_nm=300)  # broad

print(f"Laser peak @ {wl[laser_idx]:.3f} nm")
if laser: print(f"  Laser FWHM = {laser[2]:.3f} nm (left {laser[0]:.3f}, right {laser[1]:.3f})")
else:     print("  Laser FWHM not found (adjust window_nm or smoothing).")

print(f"PL peak @ {wl[pl_idx]:.3f} nm")
if pl:    print(f"  PL FWHM = {pl[2]:.3f} nm (left {pl[0]:.3f}, right {pl[1]:.3f})")
else:     print("  PL FWHM not found (adjust window_nm or smoothing).")

# ---------------- Plot (label both FWHMs) ----------------
plt.figure(figsize=(10,5), dpi=150)
plt.plot(wl, I_s, linewidth=1.2)

# Laser FWHM bar + label
if laser:
    plt.hlines(laser[5], laser[0], laser[1], linewidth=2)
    plt.text(laser[3], laser[5], f"Laser FWHM = {laser[2]:.2f} nm", ha="center", va="bottom")

# PL FWHM bar + label
if pl:
    plt.hlines(pl[5], pl[0], pl[1], linewidth=2)
    plt.text(pl[3], pl[5], f"PL FWHM = {pl[2]:.2f} nm", ha="center", va="bottom")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (counts)")
plt.title("Spectrum with Laser and PL FWHM (peak-outward crossings)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()