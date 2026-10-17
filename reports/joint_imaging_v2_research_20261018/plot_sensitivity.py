"""Render the saved readout diagnostic; requires numpy and matplotlib only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
results = json.loads((OUT / "CALIBRATION_SENSITIVITY.json").read_text(encoding="utf-8"))
main = next(p for p in results["points"] if p["temperature_K"] == 3000 and p["mean_800nm_e"] == 4000)
responses = main["gain_responses"]
labels = [r["case"] for r in responses]
t_bias = np.array([r["temperature_bias_K"] for r in responses])
e_bias = np.array([r["visible_excess_bias_e"] for r in responses])
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1, 1.35]}, layout="constrained")
y = np.arange(len(labels))
axes[0].barh(y, t_bias, color=["#ca4e3b" if v > 0 else "#316ea5" for v in t_bias])
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].axvline(0, color="#666666", linewidth=.7)
axes[0].set_xlim(-35, 56)
axes[0].set_xlabel("Color-temperature error (K)")
axes[0].set_title("Differential NIR gain changes temperature")
for i, v in enumerate(t_bias):
    axes[0].text(v + (1.1 if v >= 0 else -1.1), i, f"{v:.1f}", va="center", ha="left" if v >= 0 else "right")
limit = np.max(np.abs(e_bias))
im = axes[1].imshow(e_bias, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
axes[1].set_xticks(range(3), ["431.5 nm", "486 nm", "516.5 nm"])
axes[1].set_yticks(y, labels)
axes[1].set_title("Signed false excess after continuum subtraction")
for row in range(6):
    for col in range(3):
        v = e_bias[row, col]
        axes[1].text(col, row, f"{v:+.1f}", ha="center", va="center", color="white" if abs(v) > .55 * limit else "black")
fig.colorbar(im, ax=axes[1], label="Electrons (e)", shrink=.9)
fig.suptitle("Readout-only calculation: 3000 K, I800 = 4000 e, zero injected emission", fontsize=12)
fig.savefig(OUT / "calibration_sensitivity.png", dpi=180)
plt.close(fig)
