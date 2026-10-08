"""Thumbnail image for the write-up: figures/thumbnail.png (1280 x 720).

Alt text: "Title card reading 'Combine tracking to NFL performance: which movement measures hold up'.
Two bars compare how well first-step measures repeat between two attempts: 0.04 to 0.39 for
measures that need a detected start frame, 0.33 to 0.68 for measures that do not."

Run: python thumbnail.py   (reads work/reliability.csv from reliability.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

rel = pd.read_csv(HERE / "work" / "reliability.csv")
rel = rel[((rel.side == "OL") & (rel.drill == "PASS_RUSH_DROP")) | ((rel.side == "DL") & (rel.drill == "PASS_RUSH"))]
start_based = rel[rel.feature.isin(["speed_0p5s", "burst_dist_1s", "peak_accel_1s", "turn_rate"])].attempt_corr
start_free = rel[rel.feature.isin(["peak_speed", "best_net_1s", "best_gain_0p5s"])].attempt_corr
lo1, hi1, lo2, hi2 = start_based.min(), start_based.max(), start_free.min(), start_free.max()

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
fig.text(0.06, 0.84, "Combine tracking to NFL performance", fontsize=38, fontweight="bold", color=INK)
fig.text(0.06, 0.755, "Which movement measures hold up?", fontsize=28, color=MUTED)

ax = fig.add_axes([0.33, 0.2, 0.60, 0.38])
ax.barh([1, 0], [hi1 - lo1, hi2 - lo2], left=[lo1, lo2], height=0.5, color=[ORANGE, BLUE])
ax.set_xlim(0, 0.9)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([1, 0])
ax.set_yticklabels(["Needs a detected\nstart frame", "Does not need\none"], fontsize=20, color=INK)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(axis="x", labelsize=16, colors=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color="#dcdcd8")
ax.set_axisbelow(True)
ax.text(hi1 + 0.012, 1, f"{lo1:.2f} to {hi1:.2f}", va="center", fontsize=20, color=INK)
ax.text(hi2 + 0.012, 0, f"{lo2:.2f} to {hi2:.2f}", va="center", fontsize=20, color=INK)
fig.text(0.06, 0.09, "Repeat correlation between two attempts, range across first-step measures", fontsize=18, color=MUTED)
fig.savefig(HERE / "figures" / "thumbnail.png", facecolor="white")
print("wrote figures/thumbnail.png", f"{lo1:.2f}-{hi1:.2f}", f"{lo2:.2f}-{hi2:.2f}")
