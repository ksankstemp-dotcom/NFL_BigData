"""Robustness checks on the OL lead: best_net_1s (best distance in any one second of the OL
drills) against pass-blocking pressure allowed, with weight, height and tackle share
partialled out.

Checks, all listed before running:
  1. Other outcome definitions: logistic-adjusted pressure rate (the original), raw pressure
     rate, sack-allowed rate, mean time to pressure on plays where pressure was allowed.
  2. Team effects: drop each team in turn; and subtract each team's mean outcome.
  3. Feature variants: the lead in each OL drill separately; a second distance measure.
  4. Play-count cutoffs: 50, 100, 200, 300 pass-block plays.
  5. Permutation: Freedman-Lane (permute the outcome after removing the controls), giving the
     chance of a correlation this large for this one test, and for the largest of the six OL
     tests (3 features x 2 outcomes) taken together.

Run: python robustness_ol.py   (needs the work/*.parquet files from the earlier scripts)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from stats_utils import _residualize, bootstrap_partial, partial_rank_corr

HERE = Path(__file__).parent
RNG = np.random.default_rng(1)
CTRL = ["weight", "height", "tackle_share"]
FEATS = ["best_net_1s", "best_gain_0p5s", "peak_speed_s"]

con = duckdb.connect()
plays = con.execute("""
SELECT nfl_id, team_abbr, lined_up_position AS pos, pressure_allowed::INT AS pressure,
       sack_allowed::INT AS sack, time_to_pressure_allowed AS ttp, peak_pressure_probability_allowed AS ppp
FROM 'work/player_play.parquet'
WHERE play_nullified_by_penalty = 'N' AND pressure_allowed IS NOT NULL
""").df()
print("OL plays:", len(plays), "| sack_allowed recorded:", int(plays.sack.notna().sum()),
      "| time_to_pressure_allowed recorded:", int(plays.ttp.notna().sum()))

team = (plays.groupby(["nfl_id", "team_abbr"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).drop_duplicates("nfl_id")[["nfl_id", "team_abbr"]])
per = plays.groupby("nfl_id").agg(plays=("pressure", "size"), raw_pressure=("pressure", "mean"),
                                  sack_rate=("sack", "mean"), mean_ttp=("ttp", "mean"),
                                  n_ttp=("ttp", "count")).reset_index()
adj = pd.read_parquet(HERE / "work" / "ol_outcomes.parquet")[["nfl_id", "ol_pressure_rate_over_exp",
                                                              "ol_pressure_prob_over_exp", "tackle_share"]]
comb = con.execute("SELECT nfl_id, combine_position AS cpos, combine_weight AS weight, combine_height AS height "
                   "FROM 'work/combine_results.parquet'").df()
tf = pd.read_parquet(HERE / "work" / "trench_features.parquet")
tf = tf[tf.side == "OL"].copy()
ALLF = FEATS + ["best_path_1s_s"]
tf[ALLF] = tf.groupby("drill")[ALLF].transform(lambda s: (s - s.mean()) / s.std())
pooled = tf.groupby("nfl_id")[ALLF].mean().reset_index()

base = (per.merge(adj, on="nfl_id", how="inner").merge(team, on="nfl_id").merge(comb[comb.cpos == "OL"], on="nfl_id")
        .merge(pooled, on="nfl_id", how="left"))
print("OL players with outcome, combine and drill features:", int(base.dropna(subset=["best_net_1s"]).shape[0]))


def rho(d: pd.DataFrame, ft: str, oc: str, boot: bool = False):
    dd = d.dropna(subset=[ft, oc] + CTRL)
    if len(dd) < 20:
        return None
    x, y, Z = dd[ft].to_numpy(float), dd[oc].to_numpy(float), dd[CTRL].to_numpy(float)
    if boot:
        r = bootstrap_partial(x, y, Z, RNG, n_boot=1000)
        return len(dd), r["rho"], r["ci_lo"], r["ci_hi"]
    return len(dd), partial_rank_corr(x, y, Z), np.nan, np.nan


MAIN = base[base.plays >= 100]
LEAD = ("best_net_1s", "ol_pressure_rate_over_exp")
print("\n0. Original lead:", [round(v, 3) for v in rho(MAIN, *LEAD, boot=True)])

print("\n1. Other outcome definitions (best_net_1s):")
for oc in ["ol_pressure_rate_over_exp", "ol_pressure_prob_over_exp", "raw_pressure", "sack_rate"]:
    print(f"   {oc:28s} n,rho,ci:", [round(v, 3) for v in rho(MAIN, "best_net_1s", oc, boot=True)])
t = MAIN[MAIN.n_ttp >= 20]
r = rho(t, "best_net_1s", "mean_ttp", boot=True)
print(f"   mean_ttp (>=20 pressured plays) n,rho,ci:", [round(v, 3) for v in r] if r else "too few players")

print("\n2. Team effects:")
teams = MAIN.team_abbr.value_counts()
res = []
for tm in teams.index:
    r = rho(MAIN[MAIN.team_abbr != tm], *LEAD)
    res.append((tm, r[0], r[1]))
rs = np.array([x[2] for x in res])
print(f"   leave-one-team-out over {len(res)} teams: rho min {rs.min():.3f}, max {rs.max():.3f}, "
      f"share still negative {np.mean(rs < 0):.2f}")
worst = sorted(res, key=lambda z: z[2])[-3:]
print("   weakest when dropped:", [(a, b, round(c, 3)) for a, b, c in worst])
m2 = MAIN[MAIN.team_abbr.map(teams) >= 2].copy()
m2["oc_demeaned"] = m2[LEAD[1]] - m2.groupby("team_abbr")[LEAD[1]].transform("mean")
print("   team-mean removed (teams with >=2 players) n,rho,ci:",
      [round(v, 3) for v in rho(m2, LEAD[0], "oc_demeaned", boot=True)])

print("\n3. Feature variants:")
for dname, g in tf.groupby("drill"):
    g = g.groupby("nfl_id")[["best_net_1s"]].mean().reset_index().rename(columns={"best_net_1s": "f"})
    d = MAIN.drop(columns=["best_net_1s"]).merge(g, on="nfl_id")
    print(f"   best_net_1s in {dname:16s} n,rho,ci:", [round(v, 3) for v in rho(d, "f", LEAD[1], boot=True)])
print("   best_path_1s_s (file's own speed) n,rho,ci:", [round(v, 3) for v in rho(MAIN, "best_path_1s_s", LEAD[1], boot=True)])

print("\n4. Play-count cutoffs:")
for mn in (50, 100, 200, 300):
    r = rho(base[base.plays >= mn], *LEAD, boot=True)
    print(f"   >= {mn:3d} plays  n,rho,ci:", [round(v, 3) for v in r])

print("\n5. Permutation (Freedman-Lane, 5000 shuffles):")
d = MAIN.dropna(subset=FEATS + CTRL + ["ol_pressure_rate_over_exp", "ol_pressure_prob_over_exp"]).reset_index(drop=True)
Zr = np.column_stack([rankdata(d[c]) for c in CTRL])
Z1 = np.column_stack([np.ones(len(d)), Zr])
X = np.column_stack([_residualize(rankdata(d[f]), Zr) for f in FEATS])
Y = np.column_stack([_residualize(rankdata(d[o]), Zr) for o in ["ol_pressure_rate_over_exp", "ol_pressure_prob_over_exp"]])
Xn, Yn = X / np.linalg.norm(X, axis=0), Y / np.linalg.norm(Y, axis=0)
obs = (Xn.T @ Yn)
lead_obs = obs[0, 0]
mx_obs = np.abs(obs).max()
cnt_single = cnt_max = 0
N = 5000
for _ in range(N):
    p = RNG.permutation(len(d))
    c = Xn.T @ Yn[p]
    cnt_single += abs(c[0, 0]) >= abs(lead_obs)
    cnt_max += np.abs(c).max() >= mx_obs
print(f"   n={len(d)}; lead rho {lead_obs:.3f}; chance of |rho| this large for this one test: {(cnt_single + 1) / (N + 1):.3f}")
print(f"   largest |rho| over the 6 OL tests {mx_obs:.3f}; chance the largest of 6 is this big: {(cnt_max + 1) / (N + 1):.3f}")
