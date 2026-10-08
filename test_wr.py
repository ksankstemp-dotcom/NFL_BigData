"""WR test: do combine route-drill features relate to NFL separation over expected?

Outcome: sep_over_exp (outcomes.py), WRs with >= 30 pass-forward routes.
Predictors: stopwatch/size numbers from combine_results (baseline) and sensor features
from the sharp-break route drills (features.py).

1. Spearman correlation of every predictor with the outcome, with a bootstrap 95% interval
   and a Benjamini-Hochberg adjusted p-value across all predictors tested.
2. Repeated 5-fold cross-validated ridge regression: baseline predictors alone versus
   baseline plus sensor features, on the same players.

Run: python test_wr.py   (needs work/*.parquet from load.py, features.py, outcomes.py)
"""
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
RNG = np.random.default_rng(0)
SHARP = ("CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT")
BASELINE = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "weight"]

con = duckdb.connect()
out = con.execute(
    "SELECT nfl_id, pass_fwd, sep_over_exp FROM 'work/wr_player_outcomes.parquet' WHERE nfl_position = 'WR' ORDER BY nfl_id"
).df()
comb = con.execute(
    "SELECT nfl_id, forty, ten_yd_split, three_cone, short_shuttle, vertical, broad_jump, "
    "combine_weight AS weight FROM 'work/combine_results.parquet' ORDER BY nfl_id"
).df()
feat = pd.read_parquet(HERE / "work" / "route_features.parquet")
feat = feat[(feat.drill_type == "SKILL_DRILLS_WR") & feat.drill_name.isin(SHARP)]
sens = feat.groupby("nfl_id").agg(
    sharp_speed_lost=("speed_lost", "mean"),
    sharp_reaccel_s=("reaccel_s", "mean"),
    sharp_break_angle=("break_angle", "mean"),
    sharp_entry_speed=("entry_speed", "mean"),
    sharp_attempts=("event_id", "count"),
).reset_index()
peak = (pd.read_parquet(HERE / "work" / "route_features.parquet")
        .query("drill_type == 'SKILL_DRILLS_WR'").groupby("nfl_id").peak_speed.max()
        .rename("wr_peak_speed").reset_index())
df = out.merge(comb, on="nfl_id", how="left").merge(sens, on="nfl_id", how="left").merge(peak, on="nfl_id", how="left")
SENSOR = ["sharp_speed_lost", "sharp_reaccel_s", "sharp_break_angle", "sharp_entry_speed", "wr_peak_speed"]
print(f"WRs with outcome: {len(df)}")
print("non-missing per predictor:", df[BASELINE + SENSOR].notna().sum().to_dict())

# 1. correlations
rows = []
for c in BASELINE + SENSOR:
    d = df[[c, "sep_over_exp"]].dropna()
    r = spearmanr(d[c], d.sep_over_exp)
    boots = []
    for _ in range(2000):
        s = d.sample(len(d), replace=True, random_state=int(RNG.integers(1 << 31)))
        boots.append(spearmanr(s[c], s.sep_over_exp)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows.append({"predictor": c, "n": len(d), "rho": r[0], "ci_lo": lo, "ci_hi": hi, "p": r[1]})
res = pd.DataFrame(rows)
m = len(res)
srt = res.p.sort_values()
adj = (srt * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
res["p_bh"] = adj.reindex(res.index)
print("\nSpearman vs sep_over_exp (95% bootstrap CI):")
print(res.round(3).to_string(index=False))

# 2. cross-validated comparison on the same players
def cv_r2(cols: list[str], data: pd.DataFrame) -> tuple[float, float]:
    X, y = data[cols].to_numpy(dtype=float), data.sep_over_exp.to_numpy(dtype=float)
    scores = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=20, random_state=0).split(X):
        med = np.nanmedian(X[tr], axis=0)
        Xtr, Xte = np.where(np.isnan(X[tr]), med, X[tr]), np.where(np.isnan(X[te]), med, X[te])
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(Xtr, y[tr])
        pred = mdl.predict(Xte)
        scores.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum())
    return float(np.mean(scores)), float(np.std(scores))

need = ["forty", "vertical", "broad_jump", "weight"]   # 3-cone and shuttle exist for only 27 and 34 WRs
sub = df.dropna(subset=need + ["sharp_speed_lost", "sharp_reaccel_s"]).copy()
print(f"\nCross-validated R^2 (5-fold x 20) on {len(sub)} WRs with all of {need} and sensor features:")
for name, cols in [("baseline: " + "+".join(need), need),
                   ("baseline + sensor", need + ["sharp_speed_lost", "sharp_reaccel_s", "sharp_entry_speed"]),
                   ("sensor only", ["sharp_speed_lost", "sharp_reaccel_s", "sharp_entry_speed"])]:
    mu, sd = cv_r2(cols, sub)
    print(f"  {name:55s} R2 = {mu:+.3f}  (sd across folds {sd:.3f})")
