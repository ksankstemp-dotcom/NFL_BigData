"""DB coverage pairs from game tracking, and a separation-allowed outcome.

The game tracking holds only the 497 draft-class players (usually 1 to 3 per play, no
opponents, no ball). So a DB's separation allowed can be measured only on plays where the
receiver he faces is also a draft-class player. A pair is kept when:
  - the DB lined up at CB or SLOT_CB and was in man coverage at the snap (coverage_assignment_at_snap),
  - the receiver ran a route for the other team and a pass was thrown (pass_forward event),
  - both are tracked, and the DB started within SNAP_MAX yards of the receiver.
The distance at the snap is known before the play develops, so choosing on it does not
select on how the play went. Whether this DB was really assigned to this receiver is not
known. As a check (not used to choose pairs), the share of pairs where the DB-receiver
distance at pass_forward equals the file's own separation_at_pass_forward (nearest defender)
is reported.

Outcome per pair: distance DB to receiver at pass_forward, in yards. Expected value: the
mean for the same route type, so the outcome is the pair distance minus that mean.

Run: python db_pairs.py   (needs work/*.parquet from load.py)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SNAP_MAX = 4.0

con = duckdb.connect()
con.execute("""
CREATE VIEW g AS SELECT * FROM read_parquet(['work/game_tracking_2023.parquet',
  'work/game_tracking_2024.parquet', 'work/game_tracking_2025.parquet'])""")
df = con.execute(f"""
WITH pp AS (SELECT * FROM 'work/player_play.parquet' WHERE play_nullified_by_penalty = 'N'),
db AS (SELECT game_id, play_id, nfl_id AS db_id, team_abbr AS dteam, lined_up_position AS dpos
       FROM pp WHERE lined_up_position IN ('CB', 'SLOT_CB') AND coverage_assignment_at_snap = 'MAN'),
wr AS (SELECT game_id, play_id, nfl_id AS wr_id, team_abbr AS wteam, route_ran, cushion,
              separation_at_pass_forward AS off_sep, target::INT AS target
       FROM pp WHERE route_ran IS NOT NULL),
ev AS (SELECT game_id, play_id,
              min(time) FILTER (WHERE event = 'ball_snap') AS t_snap,
              min(time) FILTER (WHERE event = 'pass_forward') AS t_pf
       FROM g GROUP BY 1, 2 HAVING min(time) FILTER (WHERE event = 'pass_forward') IS NOT NULL),
pairs AS (SELECT * FROM db JOIN wr USING (game_id, play_id) JOIN ev USING (game_id, play_id) WHERE dteam <> wteam)
SELECT p.*, d0.x AS dx0, d0.y AS dy0, w0.x AS wx0, w0.y AS wy0,
       d1.x AS dx1, d1.y AS dy1, w1.x AS wx1, w1.y AS wy1
FROM pairs p
JOIN g d0 ON d0.game_id = p.game_id AND d0.play_id = p.play_id AND d0.nfl_id = p.db_id AND d0.time = p.t_snap
JOIN g w0 ON w0.game_id = p.game_id AND w0.play_id = p.play_id AND w0.nfl_id = p.wr_id AND w0.time = p.t_snap
JOIN g d1 ON d1.game_id = p.game_id AND d1.play_id = p.play_id AND d1.nfl_id = p.db_id AND d1.time = p.t_pf
JOIN g w1 ON w1.game_id = p.game_id AND w1.play_id = p.play_id AND w1.nfl_id = p.wr_id AND w1.time = p.t_pf
""").df()
df["dist_snap"] = np.hypot(df.dx0 - df.wx0, df.dy0 - df.wy0)
df["dist_pf"] = np.hypot(df.dx1 - df.wx1, df.dy1 - df.wy1)
print("man-coverage DB/receiver pairs with both tracked at snap and pass_forward:", len(df))

kept = df[df.dist_snap <= SNAP_MAX].copy()
print(f"kept (DB within {SNAP_MAX} yd of the receiver at the snap): {len(kept)}; "
      f"DBs: {kept.db_id.nunique()}; receivers: {kept.wr_id.nunique()}")
match = (kept.dist_pf - kept.off_sep).abs() < 0.15
print(f"check: DB-receiver distance equals the file's nearest-defender separation on {match.mean():.0%} of kept pairs "
      f"(on {(df[df.dist_snap > SNAP_MAX].pipe(lambda d: ((d.dist_pf - d.off_sep).abs() < 0.15).mean())):.0%} of the dropped ones)")

kept["exp"] = kept.groupby("route_ran").dist_pf.transform("mean")
kept["sep_allowed_over_exp"] = kept.dist_pf - kept.exp
kept.to_parquet(HERE / "work" / "db_pairs.parquet")

print("\nmean separation by route (yards):")
print(kept.groupby("route_ran").dist_pf.agg(["size", "mean"]).round(2).sort_values("size", ascending=False).to_string())

per = kept.groupby("db_id").agg(pairs=("dist_pf", "size"), sep_over_exp=("sep_allowed_over_exp", "mean")).reset_index()
print("\npairs per DB:", per.pairs.describe().round(1).to_dict())
print("DBs with >= 10 / 20 / 30 pairs:", [(per.pairs >= k).sum() for k in (10, 20, 30)])

# Split-half reliability of the outcome: odd vs even plays within each DB
kept = kept.sort_values(["db_id", "game_id", "play_id", "wr_id"])
kept["k"] = kept.groupby("db_id").cumcount()
for mn in (10, 20):
    ok = per[per.pairs >= mn].db_id
    h = kept[kept.db_id.isin(ok)].groupby(["db_id", kept.k % 2]).sep_allowed_over_exp.mean().unstack()
    r = h.corr(method="spearman").iloc[0, 1]
    print(f"split-half (odd vs even plays) reliability of separation allowed, DBs with >= {mn} pairs "
          f"(n={len(h)}): rho {r:.3f}, Spearman-Brown {2 * r / (1 + r):.3f}")
