import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# =========================
# Helpers
# =========================
def clean_df(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna().reset_index(drop=True)
    return df

def parse_bias(col, key):
    m = re.search(rf"{key} = ([\-0-9.]+)V", col)
    if m:
        return float(m.group(1))
    m = re.search(rf"{key} = ([\-0-9.]+)mV", col)
    if m:
        return float(m.group(1)) / 1000
    return None

# =========================
# Extraction functions
# =========================
def vt_gm_method(df):
    vgs = df.iloc[:, 0].astype(float).values
    results = {}

    for col in df.columns[1:]:
        vds = parse_bias(col, "Vds")
        ids_raw = df[col].astype(float).values
        ids = np.abs(ids_raw)
        vg = np.abs(vgs) if np.mean(ids_raw) < 0 else vgs

        gm = np.gradient(ids, vg)
        i = np.argmax(gm)

        vt = vg[i] - ids[i] / gm[i]

        results[vds] = {
            "Vg": vg,
            "Ids": ids,
            "gm": gm,
            "Vt": vt,
            "gm_peak": gm[i],
            "idx_peak": i
        }

    return results

# =========================
# Load data
# =========================
nfd_t = clean_df("lab05/fdsoi_and_finfet/fdsoi_and_finfet/nfdsoi_lp02_wp43_n40_transfer.csv")
nfin_t = clean_df("lab05/fdsoi_and_finfet/fdsoi_and_finfet/nfinfet_3nm_nf02_n256_transfer.csv")
nfd_o = clean_df("lab05/fdsoi_and_finfet/fdsoi_and_finfet/nfdsoi_lp02_wp43_n40_output.csv")
nfin_o = clean_df("lab05/fdsoi_and_finfet/fdsoi_and_finfet/nfinfet_3nm_nf02_n256_output.csv")

# =========================
# Extract parameters
# =========================
nfd = vt_gm_method(nfd_t)
nfin = vt_gm_method(nfin_t)

# =========================
# === PLOT 1: IDS vs VGS + Vt0 ===
# =========================
plt.figure()

# FDSOI
vg_fd = nfd[0.05]["Vg"]
ids_fd = nfd[0.05]["Ids"]
vt_fd = nfd[0.05]["Vt"]

plt.plot(vg_fd, ids_fd, label="n-FDSOI (VDS = 0.05 V)")
plt.axvline(vt_fd, linestyle='--', label=f"FDSOI Vt0 = {vt_fd:.3f} V")

# FinFET
vg_fin = nfin[0.05]["Vg"]
ids_fin = nfin[0.05]["Ids"]
vt_fin = nfin[0.05]["Vt"]

plt.plot(vg_fin, ids_fin, label="n-FinFET (VDS = 0.05 V)")
plt.axvline(vt_fin, linestyle='--', label=f"FinFET Vt0 = {vt_fin:.3f} V")

plt.title("Transfer Characteristics with Extracted Threshold Voltage")
plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("Drain Current IDS (A)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# =========================
# === PLOT 2: gm vs VGS + Vt0 ===
# =========================
plt.figure()

gm_fd = nfd[0.05]["gm"]
gm_fin = nfin[0.05]["gm"]

plt.plot(vg_fd, gm_fd, label="n-FDSOI gm")
plt.plot(vg_fin, gm_fin, label="n-FinFET gm")

plt.axvline(vt_fd, linestyle='--', label=f"FDSOI Vt0")
plt.axvline(vt_fin, linestyle='--', label=f"FinFET Vt0")

plt.title("Transconductance vs Gate Voltage")
plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("Transconductance gm (S)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# =========================
# === PLOT 3: LOG IDS (Subthreshold) ===
# =========================
plt.figure()

plt.semilogy(vg_fd, ids_fd, label="n-FDSOI")
plt.semilogy(vg_fin, ids_fin, label="n-FinFET")

plt.title("Subthreshold Characteristics (Log Scale)")
plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("Drain Current IDS (A)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()

# =========================
# === PLOT 4: OUTPUT (n-FDSOI) ===
# =========================
plt.figure()

vds = nfd_o.iloc[:, 0]

for col in nfd_o.columns[1:]:
    vgs = parse_bias(col, "Vgs")
    ids = np.abs(nfd_o[col])
    plt.plot(vds, ids, label=f"VGS = {vgs:.2f} V")

plt.title("Output Characteristics: n-FDSOI")
plt.xlabel("Drain Voltage VDS (V)")
plt.ylabel("Drain Current IDS (A)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# =========================
# === PLOT 5: OUTPUT (n-FinFET) ===
# =========================
plt.figure()

vds = nfin_o.iloc[:, 0]

for col in nfin_o.columns[1:]:
    vgs = parse_bias(col, "Vgs")
    ids = np.abs(nfin_o[col])
    plt.plot(vds, ids, label=f"VGS = {vgs:.2f} V")

plt.title("Output Characteristics: n-FinFET")
plt.xlabel("Drain Voltage VDS (V)")
plt.ylabel("Drain Current IDS (A)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()