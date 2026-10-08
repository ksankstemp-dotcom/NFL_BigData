"""Route-matched WR test with controls.

Each NFL route is paired with the combine drill that most resembles it, and the sensor
features from that drill are tested against the player's separation over expected on that
route only. Weight, height and the share of routes run from the slot are partialled out of
both sides first (rank-transform, regress on the controls, correlate the residuals).

The pairings are my judgement of which drill looks like which route and were fixed before
looking at results: HITCH-curl/comeback, OUT-speed out, SLANT-slant, IN-dagger,
POST and CORNER-post-corner, GO-go (peak speed only).

Run: python test_wr_matched.py   (needs work/*.parquet from load.py, features.py, outcomes.py)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)
PAIRS = {
    "HITCH": ["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT"],
    "OUT": ["SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT"],
    "SLANT": ["SLANT_ROUTE_LEFT"],
    "IN": ["DAGGER_ROUTE_LEFT"],
    "POST": ["POST_CORNER_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT"],
    "CORNER": ["POST_CORNER_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT"],
    "GO": ["GO_ROUTE_RIGHT"],
}
CUT_FEATURES = ["speed_lost", "reaccel_s", "break_angle"]
GO_FEATURES = ["peak_speed"]
CONTROLS = ["weight", "height", "slot_share"]

con = duckdb.connect()
route_out = con.execute(
    "SELECT nfl_id, route_ran, sep_over_exp, pass_fwd FROM 'work/wr_route_outcomes.parquet' WHERE nfl_position = 'WR' ORDER BY nfl_id, route_ran"
).df()
ctrl = con.execute("""
SELECT c.nfl_id, c.combine_weight AS weight, c.combine_height AS height,
       s.slot_share
FROM 'work/combine_results.parquet' c
JOIN (
  SELECT nfl_id, avg((lined_up_position = 'SLOT_WR')::INT) AS slot_share
  FROM 'work/player_play.parquet'
  WHERE route_ran IS NOT NULL AND play_nullified_by_penalty = 'N'
  GROUP BY nfl_id
) s USING (nfl_id)
ORDER BY c.nfl_id
""").df()
feat = pd.read_parquet(HERE / "work" / "route_features.parquet")
feat = feat[feat.drill_type == "SKILL_DRILLS_WR"]


def residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, v, rcond=None)
    return v - Z1 @ beta


def partial_rank_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    Zr = np.column_stack([rankdata(Z[:, i]) for i in range(Z.shape[1])])
    return pearsonr(residualize(rankdata(x), Zr), residualize(rankdata(y), Zr))[0]


rows = []
for route, drills in PAIRS.items():
    f = feat[feat.drill_name.isin(drills)].groupby("nfl_id")[CUT_FEATURES + GO_FEATURES].mean().reset_index()
    d = (route_out[route_out.route_ran == route].merge(f, on="nfl_id").merge(ctrl, on="nfl_id")
         .replace([np.inf, -np.inf], np.nan))
    for ft in (GO_FEATURES if route == "GO" else CUT_FEATURES):
        dd = d.dropna(subset=[ft, "sep_over_exp"] + CONTROLS)
        if len(dd) < 20:
            continue
        x, y, Z = dd[ft].to_numpy(float), dd.sep_over_exp.to_numpy(float), dd[CONTROLS].to_numpy(float)
        raw = pearsonr(rankdata(x), rankdata(y))[0]
        part = partial_rank_corr(x, y, Z)
        boots = []
        for _ in range(2000):
            i = RNG.integers(0, len(dd), len(dd))
            boots.append(partial_rank_corr(x[i], y[i], Z[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * (np.mean(np.array(boots) <= 0) if part > 0 else np.mean(np.array(boots) >= 0))
        rows.append({"route": route, "feature": ft, "n": len(dd), "raw_rho": raw,
                     "partial_rho": part, "ci_lo": lo, "ci_hi": hi, "p_boot": min(p, 1.0)})

res = pd.DataFrame(rows)
m = len(res)
srt = res.p_boot.sort_values()
adj = (srt * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
res["p_bh"] = adj.reindex(res.index)
pd.set_option("display.width", 200)
print(f"{m} tests; controls partialled out: {CONTROLS}")
print(res.round(3).to_string(index=False))
res.to_csv(HERE / "work" / "wr_matched_results.csv", index=False)
