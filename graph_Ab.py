import numpy as np
import matplotlib.pyplot as plt
import os

# --- paths (edit if yours are elsewhere) ---
files = {
    "Sample 1": "data\GaAs Ab\Absorbance__0__10-47-44-395.txt",
    "Sample 2": "data\Si Ab\Absorbance__2__10-51-58-535.txt",
    "Sample 3": "data\Oneside Ab\Absorbance__4__10-53-28-609.txt",
}
# --- output folder ---
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

def read_absorbance_txt(path):
    """
    Reads Ocean Optics-style absorbance text file.
    Returns wavelength (nm) and absorbance (a.u.).
    """
    wl, ab = [], []
    data_start = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ">>>>>Begin Spectral Data<<<<<" in line:
                data_start = True
                continue
            if not data_start or not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                wl.append(float(parts[0]))
                ab.append(float(parts[1]))
            except ValueError:
                continue

    wl = np.array(wl)
    ab = np.array(ab)

    idx = np.argsort(wl)
    return wl[idx], ab[idx]


# --- plot & save each sample ---
for name, path in files.items():
    wavelength, absorbance = read_absorbance_txt(path)

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(wavelength, absorbance, linewidth=1.5)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (a.u.)")
    plt.title(f"{name}: Absorbance vs Wavelength")
    plt.grid(True, alpha=0.25)
    plt.xlim(600, 1300)   # adjust if needed

    filename = f"{name.replace(' ', '_')}_Absorbance.png"
    save_path = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")