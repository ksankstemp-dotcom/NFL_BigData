"""Figures for the write-up, drawn from the saved result files. Each figure also writes the
values it plots to figures/<name>.csv, which is its text version.

Run after the analysis scripts: python charts.py
Reads work/reliability.csv, work/overlap_results.csv, work/wr_matched_results.csv,
work/trench_results_original.csv, work/trench_results.csv and results/14_robustness_ol.txt.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated pair (see dataviz validator): CVD and contrast pass
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name, data: pd.DataFrame):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    data.to_csv(OUT / f"{name}.csv", index=False)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: reliability
rel = pd.read_csv(HERE / "work" / "reliability.csv")
rel = rel[(rel.side == "OL") & (rel.drill == "PASS_RUSH_DROP") | (rel.side == "DL") & (rel.drill == "PASS_RUSH")]
NAMES = {
    "speed_0p5s": "Speed 0.5 s after start", "burst_dist_1s": "Distance in 1st second after start",
    "peak_accel_1s": "Peak acceleration, 1st second", "turn_rate": "Turn rate",
    "peak_speed": "Peak speed", "best_net_1s": "Best distance in any 1 s",
    "best_gain_0p5s": "Best speed gain in any 0.5 s",
}
ORDER = ["speed_0p5s", "burst_dist_1s", "peak_accel_1s", "turn_rate",       # need a detected start
         "peak_speed", "best_net_1s", "best_gain_0p5s"]                      # do not
p = rel[rel.feature.isin(ORDER)].pivot(index="feature", columns="side", values="attempt_corr").loc[ORDER]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
y = np.arange(len(ORDER))[::-1].astype(float)
y[4:] -= 0.6   # gap between the two groups
h = 0.34
for off, side, col in [(h / 2, "DL", ORANGE), (-h / 2, "OL", BLUE)]:
    ax.barh(y + off, p[side], height=h, color=col, label=f"{'Defensive' if side == 'DL' else 'Offensive'} line")
    for yy, v in zip(y + off, p[side]):
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([NAMES[f] for f in ORDER])
ax.axvline(0.6, color=MUTED, ls="--", lw=1)
ax.text(0.605, y.max() + 0.62, "0.6: level we aimed for", fontsize=9, color=MUTED, va="bottom")
ax.set_ylim(y.min() - 0.6, y.max() + 1.25)
ax.set_xlim(0, 0.85)
ax.set_xlabel("Correlation between an athlete's first and second attempt (Spearman)")
ax.text(0.0, y[0] + 0.62, "Needs a detected start frame", fontsize=10, color=INK, fontweight="bold")
ax.text(0.0, y[4] + 0.62, "Does not need a start frame", fontsize=10, color=INK, fontweight="bold")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Measures that need a start frame barely repeat", loc="left", fontsize=13, fontweight="bold")
save(fig, "fig1_reliability", p.reset_index().rename(columns={"OL": "OL_pass_rush_drop", "DL": "DL_pass_rush"}))

# ---------------------------------------------------------------- Figure 2: overlap
ov = pd.read_csv(HERE / "work" / "overlap_results.csv")
LAB = {"speed_lost": "speed lost", "reaccel_s": "time to re-accelerate", "break_angle": "break angle",
       "entry_speed": "entry speed", "peak_speed": "peak speed", "best_net_1s": "best distance in 1 s",
       "best_gain_0p5s": "best speed gain in 0.5 s", "peak_speed_s": "peak speed (file's speed)"}
ov["label"] = ov.group.str.replace(" drills", "").str.replace("WR sharp cuts", "WR cuts") + ": " + ov.feature.map(LAB)
ov = ov.sort_values("cv_r2_stopwatch")
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.barh(ov.label, ov.cv_r2_stopwatch, height=0.62, color=BLUE)
for yy, v in zip(range(len(ov)), ov.cv_r2_stopwatch):
    ax.text(max(v, 0) + 0.01, yy, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-0.12, 0.8)
ax.set_xlabel("Share of the feature predicted by 40 time, 10-yard split, jumps, weight and height (cross-validated R-squared)")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Only defensive-line speed measures mostly repeat the stopwatch", loc="left", fontsize=13, fontweight="bold")
save(fig, "fig2_overlap", ov[["group", "feature", "n", "cv_r2_stopwatch", "rho_with_forty"]].iloc[::-1])

# ---------------------------------------------------------------- Figure 3: every test
fam = []
w = pd.read_csv(HERE / "work" / "wr_matched_results.csv").assign(family="Receivers by route (19)")
w["name"] = w.route + " " + w.feature
t1 = pd.read_csv(HERE / "work" / "trench_results_original.csv").assign(family="Linemen, first pass (20)")
t2 = pd.read_csv(HERE / "work" / "trench_results.csv").assign(family="Linemen, second pass (12)")
for d in (t1, t2):
    d["name"] = d.side + " " + d.feature + " vs " + d.outcome.str.replace("_over_exp", "").str.replace("_over_pos", "")
al = pd.concat([w, t1, t2], ignore_index=True)
al["excludes_zero"] = (al.ci_lo > 0) | (al.ci_hi < 0)
fig, ax = plt.subplots(figsize=(8.6, 6.4))
ypos, labels, ylab = 0, [], []
FAMS = ["Receivers by route (19)", "Linemen, first pass (20)", "Linemen, second pass (12)"]
for f in FAMS:
    d = al[al.family == f].sort_values("partial_rho").reset_index(drop=True)
    ys = ypos + np.arange(len(d))
    for yy, (_, r) in zip(ys, d.iterrows()):
        col = ORANGE if r.excludes_zero else BLUE
        if not np.isnan(r.ci_lo):
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.6, alpha=0.55 if not r.excludes_zero else 1)
        ax.plot(r.partial_rho, yy, "o", color=col, ms=5.5, mec="white", mew=1)
        if r.excludes_zero:
            ax.text(r.ci_hi + 0.02, yy, f"{r['name']}: p {r.p_boot:.3f}, adjusted {r.p_bh:.2f}", fontsize=8.5, va="center", color=INK)
    ax.text(-0.98, ys.max() + 1.6, f, fontsize=10, fontweight="bold", color=INK)
    ypos = ys.max() + 4
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1.5, ypos - 1.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Partial rank correlation with the NFL outcome, with 95% interval")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
n_ex = int(al.excludes_zero.sum())
ax.text(0.98, 0.02, "Blue: interval includes zero.  Orange: interval misses zero.", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
ax.set_title(f"{len(al)} movement tests: {n_ex} intervals miss zero, none survive correction", loc="left", fontsize=13, fontweight="bold")
save(fig, "fig3_all_tests", al[["family", "name", "n", "partial_rho", "ci_lo", "ci_hi", "p_boot", "p_bh", "excludes_zero"]])

# ---------------------------------------------------------------- Figure 4: robustness of the OL lead
log = (HERE / "results" / "14_robustness_ol.txt").read_text()
num = r"np\.float64\((-?[\d.]+)\)"
pat = re.compile(r"^\s{3}(.+?)\s+n,rho,ci:\s*\[(\d+), " + num + ", " + num + ", " + num + r"\]", re.M)
rows = [(m.group(1).strip(), int(m.group(2)), *map(float, m.groups()[2:])) for m in pat.finditer(log)]
pat2 = re.compile(r"^\s{3}(>= +\d+ plays)\s+n,rho,ci:\s*\[(\d+), " + num + ", " + num + ", " + num + r"\]", re.M)
rows += [(m.group(1).replace("  ", " "), int(m.group(2)), *map(float, m.groups()[2:])) for m in pat2.finditer(log)]
m3 = re.search(r"team-mean removed .*?\[(\d+), " + num + ", " + num + ", " + num + r"\]", log)
rows.append(("team-mean removed", int(m3.group(1)), *map(float, m3.groups()[1:])))
rb = pd.DataFrame(rows, columns=["check", "n", "rho", "ci_lo", "ci_hi"]).drop_duplicates("check")
KEEP = {
    "ol_pressure_rate_over_exp": "Original: pressure rate over expected",
    "ol_pressure_prob_over_exp": "Pressure probability over expected",
    "raw_pressure": "Raw pressure rate",
    "sack_rate": "Sack rate (expected +? no: lower is better)",
    "mean_ttp (>=20 pressured plays)": "Time to pressure (higher is better)",
    "team-mean removed": "Team average removed",
    "best_net_1s in PASS_PRO_MIRROR": "Mirror drill only",
    "best_net_1s in PASS_RUSH_DROP": "Pass-rush-drop drill only",
    "best_net_1s in WAVE_SLIDE": "Wave-slide drill only",
    "best_path_1s_s (file's own speed)": "Distance from file's own speed",
    ">= 200 plays": "At least 200 pass-block plays",
    ">= 300 plays": "At least 300 pass-block plays",
}
KEEP["sack_rate"] = "Sack rate"
rb = rb[rb.check.isin(KEEP)].copy()
rb["label"] = rb.check.map(KEEP)
rb = rb.set_index("check").loc[list(KEEP)].reset_index()
fig, ax = plt.subplots(figsize=(8.6, 5.6))
yy = np.arange(len(rb))[::-1]
for y_, (_, r) in zip(yy, rb.iterrows()):
    col = BLUE if r.ci_hi < 0 else MUTED
    ax.plot([r.ci_lo, r.ci_hi], [y_, y_], color=col, lw=1.8)
    ax.plot(r.rho, y_, "o", color=BLUE if r.check == "ol_pressure_rate_over_exp" else col, ms=7, mec="white", mew=1.2)
    ax.text(0.5, y_, f"{r.rho:+.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(yy)
ax.set_yticklabels(rb.label)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-0.6, 0.62)
ax.set_xlabel("Correlation of best distance in 1 s with the outcome, with 95% interval")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("The offensive-line lead fades under other definitions", loc="left", fontsize=13, fontweight="bold")
fig.text(0.02, -0.03, "Negative means more distance goes with less pressure allowed, except time to pressure, where positive is better.",
         fontsize=9, color=MUTED)
save(fig, "fig4_ol_robustness", rb[["label", "n", "rho", "ci_lo", "ci_hi"]])
print("wrote", sorted(p.name for p in OUT.iterdir()))
print("tests:", len(al), "| excluding zero:", n_ex)
print(al[al.excludes_zero][["family", "name", "partial_rho", "ci_lo", "ci_hi", "p_boot", "p_bh"]].round(3).to_string(index=False))
