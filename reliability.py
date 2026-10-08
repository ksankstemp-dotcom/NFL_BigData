"""Repeat-attempt reliability of every trench feature, per drill.

For athletes with two attempts at the same drill, the Spearman correlation between the first
and second attempt. Also the Spearman-Brown value for the mean of two attempts
(2r / (1 + r)), which is the reliability of a feature averaged over both.

Run: python reliability.py   (needs work/trench_features.parquet from trench_features.py)
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
tf = pd.read_parquet(HERE / "work" / "trench_features.parquet")
FEATS = [c for c in tf.columns if c not in ("side", "drill", "event_id", "nfl_id", "frames")]

rows = []
for (side, drill), g in tf.groupby(["side", "drill"]):
    g = g.sort_values("event_id")
    g["k"] = g.groupby("nfl_id").cumcount()
    a, b = g[g.k == 0].set_index("nfl_id"), g[g.k == 1].set_index("nfl_id")
    both = a.index.intersection(b.index)
    if len(both) < 20:
        continue
    for c in FEATS:
        r = spearmanr(a.loc[both, c], b.loc[both, c])[0]
        rows.append({"side": side, "drill": drill, "feature": c, "n": len(both),
                     "attempt_corr": r, "mean_of_two": 2 * r / (1 + r) if r > -1 else float("nan")})
res = pd.DataFrame(rows).sort_values(["side", "attempt_corr"], ascending=[True, False])
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv(HERE / "work" / "reliability.csv", index=False)
