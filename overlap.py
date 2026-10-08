"""How much of each sensor feature is already explained by the stopwatch and size numbers?

For every sensor feature (WR sharp-cut drills, OL drills, DL drills) this fits a
cross-validated ridge regression of the feature on forty, 10-yard split, vertical, broad
jump, weight and height, and reports the out-of-sample R^2 (5-fold, 20 repeats). It also
reports the Spearman correlation with the 40 alone. A feature with a low R^2 carries
information the stopwatch does not; a high R^2 means it largely repeats it.

Run: python overlap.py   (needs work/*.parquet from the earlier scripts)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
STOPWATCH = ["forty", "ten_yd_split", "vertical", "broad_jump", "combine_weight", "combine_height"]
SHARP = ("CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT")
TRENCH = ["best_net_1s", "best_gain_0p5s", "peak_speed_s"]
if "--original" in sys.argv:   # the first-pass start-based features
    TRENCH = ["speed_0p5s", "burst_dist_1s", "peak_accel_1s", "peak_speed", "turn_rate"]

con = duckdb.connect()
comb = con.execute(
    "SELECT nfl_id, combine_position AS cpos, forty, ten_yd_split, vertical, broad_jump, "
    "combine_weight, combine_height FROM 'work/combine_results.parquet'"
).df()

wr = pd.read_parquet(HERE / "work" / "route_features.parquet")
wr = wr[(wr.drill_type == "SKILL_DRILLS_WR") & wr.drill_name.isin(SHARP)]
wr = wr.groupby("nfl_id")[["speed_lost", "reaccel_s", "break_angle", "entry_speed", "peak_speed"]].mean().reset_index()

tf = pd.read_parquet(HERE / "work" / "trench_features.parquet")
tf[TRENCH] = tf.groupby(["side", "drill"])[TRENCH].transform(lambda s: (s - s.mean()) / s.std())
tr = {s: g.groupby("nfl_id")[TRENCH].mean().reset_index() for s, g in tf.groupby("side")}

GROUPS = [("WR sharp cuts", "WR", wr, list(wr.columns[1:])),
          ("OL drills", "OL", tr["OL"], TRENCH),
          ("DL drills", "DL", tr["DL"], TRENCH)]


def cv_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    scores = []
    for a, b in RepeatedKFold(n_splits=5, n_repeats=20, random_state=0).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(X[a], y[a])
        scores.append(1 - ((y[b] - m.predict(X[b])) ** 2).sum() / ((y[b] - y[a].mean()) ** 2).sum())
    return float(np.mean(scores)), float(np.std(scores))


rows = []
for label, cpos, feats, cols in GROUPS:
    d = comb[comb.cpos == cpos].merge(feats, on="nfl_id").dropna(subset=STOPWATCH)
    for c in cols:
        dd = d.dropna(subset=[c])
        X, y = dd[STOPWATCH].to_numpy(float), dd[c].to_numpy(float)
        r2, sd = cv_r2(X, y)
        rho = spearmanr(dd.forty, dd[c])[0]
        rows.append({"group": label, "feature": c, "n": len(dd), "cv_r2_stopwatch": r2,
                     "fold_sd": sd, "rho_with_forty": rho})

# Repeat-attempt reliability: correlation between an athlete's first and second attempt
# of the same drill. A feature that does not repeat cannot be explained by anything.
rel = []
for (side, drill), g in tf.groupby(["side", "drill"]):
    g = g.sort_values("event_id")
    g["k"] = g.groupby("nfl_id").cumcount()
    a, b = g[g.k == 0].set_index("nfl_id"), g[g.k == 1].set_index("nfl_id")
    both = a.index.intersection(b.index)
    for c in TRENCH:
        if len(both) >= 20:
            rel.append({"side": side, "drill": drill, "feature": c, "athletes_with_2": len(both),
                        "attempt_corr": spearmanr(a.loc[both, c], b.loc[both, c])[0]})
reliability = pd.DataFrame(rel)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"Stopwatch + size predictors: {STOPWATCH}")
print(res.round(3).to_string(index=False))
res.to_csv(HERE / "work" / "overlap_results.csv", index=False)
print("\nRepeat-attempt reliability (Spearman, first vs second attempt, same athlete and drill):")
print(reliability.round(3).to_string(index=False))
