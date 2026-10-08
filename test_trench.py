"""Trench test: do OL/DL combine drill movement features relate to NFL pass-blocking and
pass-rush results?

Sensor features (trench_features.py) are z-scored within each drill and averaged over the
player's matched drills. Each is tested against two outcomes per side (trench_outcomes.py)
with weight, height and the share of snaps at the main position (tackle for OL, edge for DL)
partialled out. The 20 sensor tests are one family for the Benjamini-Hochberg adjustment.
The stopwatch baselines (10-yard split, 40) are reported beside them, uncontrolled and
controlled, and are not part of the family.

Run: python test_trench.py   (needs work/*.parquet from the earlier scripts)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from stats_utils import bh_adjust, bootstrap_partial

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)
# Second pass: three start-free features chosen from the reliability table (reliability.py).
# `python test_trench.py --original` runs the first pass with the five start-based features.
FEATURES = ["best_net_1s", "best_gain_0p5s", "peak_speed_s"]
if "--original" in sys.argv:
    FEATURES = ["speed_0p5s", "burst_dist_1s", "peak_accel_1s", "peak_speed", "turn_rate"]
MIN_OL, MIN_DL_GETOFF, MIN_DL_RUSH = 100, 50, 100

con = duckdb.connect()
comb = con.execute(
    "SELECT nfl_id, combine_position AS cpos, combine_weight AS weight, combine_height AS height, "
    "ten_yd_split, forty FROM 'work/combine_results.parquet'"
).df()
tf = pd.read_parquet(HERE / "work" / "trench_features.parquet")
tf[FEATURES] = tf.groupby(["side", "drill"])[FEATURES].transform(lambda s: (s - s.mean()) / s.std())
pooled = tf.groupby(["side", "nfl_id"])[FEATURES].mean().reset_index()

ol_o = pd.read_parquet(HERE / "work" / "ol_outcomes.parquet")
dl_o = pd.read_parquet(HERE / "work" / "dl_outcomes.parquet")
SIDES = {
    "OL": dict(out=ol_o.query("ol_plays >= @MIN_OL"), cpos="OL", share="tackle_share",
               outcomes=["ol_pressure_rate_over_exp", "ol_pressure_prob_over_exp"]),
    "DL": dict(out=dl_o, cpos="DL", share="edge_share",
               outcomes=["dl_get_off_over_pos", "dl_pressure_rate_over_pos"]),
}
MINS = {"dl_get_off_over_pos": ("dl_getoff_plays", MIN_DL_GETOFF),
        "dl_pressure_rate_over_pos": ("dl_rush_plays", MIN_DL_RUSH)}

rows, base = [], []
for side, cfg in SIDES.items():
    d0 = (cfg["out"].merge(comb[comb.cpos == cfg["cpos"]], on="nfl_id")
          .merge(pooled[pooled.side == side].drop(columns="side"), on="nfl_id", how="left"))
    ctrl_cols = ["weight", "height", cfg["share"]]
    for oc in cfg["outcomes"]:
        d1 = d0
        if oc in MINS:
            col, mn = MINS[oc]
            d1 = d0[d0[col] >= mn]
        for ft in FEATURES + ["ten_yd_split", "forty"]:
            dd = d1.dropna(subset=[ft, oc] + ctrl_cols)
            if len(dd) < 25:
                continue
            x, y, Z = dd[ft].to_numpy(float), dd[oc].to_numpy(float), dd[ctrl_cols].to_numpy(float)
            r = bootstrap_partial(x, y, Z, RNG)
            r_raw = bootstrap_partial(x, y, None, RNG) if ft in ("ten_yd_split", "forty") else {"rho": np.nan}
            rec = {"side": side, "outcome": oc, "feature": ft, "n": len(dd), "raw_rho": r_raw["rho"],
                   "partial_rho": r["rho"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_boot": r["p_boot"]}
            (base if ft in ("ten_yd_split", "forty") else rows).append(rec)

res = pd.DataFrame(rows)
res["p_bh"] = bh_adjust(res.p_boot)
pd.set_option("display.width", 220)
print(f"Sensor features: {len(res)} tests (one family)")
print(res.drop(columns="raw_rho").round(3).to_string(index=False))
print("\nStopwatch baselines (not in the family):")
print(pd.DataFrame(base).round(3).to_string(index=False))
res.to_csv(HERE / "work" / ("trench_results_original.csv" if "--original" in sys.argv else "trench_results.csv"), index=False)
