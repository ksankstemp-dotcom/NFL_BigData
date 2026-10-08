"""Replication check across draft classes for the two leads from test_trench.py.

Fixed in advance, and only these two pairs are tested:
  DL: best_gain_0p5s  vs dl_pressure_rate_over_pos   (>= 100 pass-rush plays)
  OL: best_net_1s     vs ol_pressure_rate_over_exp   (>= 100 pass-block plays)
Same controls as before (weight, height, share of snaps at the main position). For each draft
class the partial rank correlation and a bootstrap interval; a lead counts as replicating
only if the sign matches in all three classes. Exploratory either way.

Run: python replication.py   (needs the work/*.parquet files from the earlier scripts)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from stats_utils import bootstrap_partial

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)
FEATS = ["best_net_1s", "best_gain_0p5s"]

con = duckdb.connect()
comb = con.execute(
    "SELECT nfl_id, draft_year, combine_position AS cpos, combine_weight AS weight, combine_height AS height "
    "FROM 'work/combine_results.parquet'"
).df()
tf = pd.read_parquet(HERE / "work" / "trench_features.parquet")
tf[FEATS] = tf.groupby(["side", "drill"])[FEATS].transform(lambda s: (s - s.mean()) / s.std())
pooled = tf.groupby(["side", "nfl_id"])[FEATS].mean().reset_index()
ol = pd.read_parquet(HERE / "work" / "ol_outcomes.parquet").query("ol_plays >= 100")
dl = pd.read_parquet(HERE / "work" / "dl_outcomes.parquet").query("dl_rush_plays >= 100")

LEADS = [("DL", "best_gain_0p5s", "dl_pressure_rate_over_pos", dl, "edge_share"),
         ("OL", "best_net_1s", "ol_pressure_rate_over_exp", ol, "tackle_share")]
rows = []
for side, ft, oc, out, share in LEADS:
    d = (out.merge(comb[comb.cpos == side], on="nfl_id")
         .merge(pooled[pooled.side == side].drop(columns="side"), on="nfl_id")
         .dropna(subset=[ft, oc, "weight", "height", share]))
    ctrl = ["weight", "height", share]
    groups = [("all", d)] + [(str(y), g) for y, g in d.groupby("draft_year")]
    for label, g in groups:
        if len(g) < 12:
            continue
        r = bootstrap_partial(g[ft].to_numpy(float), g[oc].to_numpy(float), g[ctrl].to_numpy(float), RNG)
        rows.append({"lead": f"{side} {ft} -> {oc}", "class": label, "n": len(g),
                     "partial_rho": r["rho"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
for lead, g in res.groupby("lead", sort=False):
    cls = g[g["class"] != "all"]
    same = (np.sign(cls.partial_rho) == np.sign(g[g["class"] == "all"].partial_rho.iloc[0])).all()
    print(f"{lead}: sign matches pooled in all {len(cls)} classes: {same}")
