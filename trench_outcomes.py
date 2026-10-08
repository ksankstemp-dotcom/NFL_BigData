"""NFL trench outcomes from player_play, one row per player.

OL (pass blocking). Rows where pressure_allowed is recorded.
  ol_pressure_rate_over_exp   pressure_allowed minus the league-wide logistic prediction from
                              dropback_duration and pass_rushers_encountered (longer drop-backs
                              and more rushers allow more pressure whoever is blocking).
  ol_pressure_prob_over_exp   the same for peak_pressure_probability_allowed, a continuous
                              measure that is less noisy than a yes/no pressure.

DL (pass rush). Rows where the defender lined up as EDGE or INTERIOR_LINE with a recorded
get-off or pass-rush result.
  dl_get_off_over_pos         player_get_off (a time in seconds, 0 to 2; pressured plays
                              average 0.88 against 1.02 for the rest, so lower is probably
                              faster) minus the average for the same lined-up position.
  dl_pressure_rate_over_pos   whether the rusher got pressure (time_to_pressure recorded) per
                              pass-rush play, minus the average for the same position.

Plays voided by a penalty are excluded. The meaning of get-off and of the pass-rush
denominator are inferred from the data, not from a data dictionary.

Run: python trench_outcomes.py   (needs work/*.parquet from load.py)
"""
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

HERE = Path(__file__).parent
MIN_OL = 100
MIN_DL_GETOFF = 50
MIN_DL_RUSH = 100

con = duckdb.connect()
ol = con.execute("""
SELECT nfl_id, lined_up_position AS pos, pressure_allowed::INT AS pressure,
       peak_pressure_probability_allowed AS ppp, dropback_duration AS drop, pass_rushers_encountered AS rushers
FROM 'work/player_play.parquet'
WHERE play_nullified_by_penalty = 'N' AND pressure_allowed IS NOT NULL
  AND dropback_duration IS NOT NULL AND pass_rushers_encountered IS NOT NULL
""").df()
X = ol[["drop", "rushers"]].to_numpy(float)
ol["pressure_exp"] = LogisticRegression(max_iter=1000).fit(X, ol.pressure).predict_proba(X)[:, 1]
has = ol.ppp.notna()
ol["ppp_exp"] = np.nan
ol.loc[has, "ppp_exp"] = LinearRegression().fit(X[has], ol.ppp[has]).predict(X[has])
ol["tackle"] = (ol.pos == "T").astype(int)
ol_p = (ol.groupby("nfl_id").agg(
            ol_plays=("pressure", "size"),
            ol_pressure_rate=("pressure", "mean"),
            ol_pressure_rate_over_exp=("pressure", lambda s: float((s - ol.loc[s.index, "pressure_exp"]).mean())),
            ol_pressure_prob_over_exp=("ppp", lambda s: float((s - ol.loc[s.index, "ppp_exp"]).mean())),
            tackle_share=("tackle", "mean"))
        .query("ol_plays >= @MIN_OL").reset_index())
ol_p.to_parquet(HERE / "work" / "ol_outcomes.parquet")

dl = con.execute("""
SELECT nfl_id, lined_up_position AS pos, player_get_off AS getoff,
       CASE WHEN sack IS NOT NULL THEN (time_to_pressure IS NOT NULL)::INT END AS pressure
FROM 'work/player_play.parquet'
WHERE play_nullified_by_penalty = 'N' AND lined_up_position IN ('EDGE', 'INTERIOR_LINE')
""").df()
dl["pressure"] = dl.pressure.astype(float)
dl["getoff"] = dl.getoff.astype(float)
dl["getoff_exp"] =dl.groupby("pos").getoff.transform("mean")
dl["pressure_exp"] = dl.groupby("pos").pressure.transform("mean")
dl["edge"] = (dl.pos == "EDGE").astype(int)
dl_p = (dl.groupby("nfl_id").agg(
            dl_getoff_plays=("getoff", "count"),
            dl_rush_plays=("pressure", "count"),
            dl_get_off=("getoff", "mean"),
            dl_get_off_over_pos=("getoff", lambda s: float((s - dl.loc[s.index, "getoff_exp"]).mean())),
            dl_pressure_rate=("pressure", "mean"),
            dl_pressure_rate_over_pos=("pressure", lambda s: float((s - dl.loc[s.index, "pressure_exp"]).mean())),
            edge_share=("edge", "mean"))
        .reset_index())
dl_p.to_parquet(HERE / "work" / "dl_outcomes.parquet")

print(f"OL players with >= {MIN_OL} pass-block plays: {len(ol_p)}")
print(ol_p[["ol_plays", "ol_pressure_rate", "ol_pressure_rate_over_exp", "ol_pressure_prob_over_exp"]].describe().round(3).to_string())
print(f"\nDL players: get-off >= {MIN_DL_GETOFF}: {(dl_p.dl_getoff_plays >= MIN_DL_GETOFF).sum()}, "
      f"rush plays >= {MIN_DL_RUSH}: {(dl_p.dl_rush_plays >= MIN_DL_RUSH).sum()}")
print(dl_p[["dl_getoff_plays", "dl_rush_plays", "dl_get_off", "dl_get_off_over_pos", "dl_pressure_rate",
            "dl_pressure_rate_over_pos"]].describe().round(3).to_string())
