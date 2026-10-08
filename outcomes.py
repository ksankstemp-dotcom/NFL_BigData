"""NFL receiver outcomes from player_play, one row per player (and per player and route).

Separation is only recorded when a pass was thrown, so it is a measure of getting open on
routes where the quarterback let go of the ball. It depends heavily on route type and on
man versus zone coverage, so each play's separation is compared with the average for the
same route and coverage type ("separation over expected"). Plays voided by a penalty are
excluded.

Run: python outcomes.py   (needs work/*.parquet from load.py)
"""
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
MIN_PLAYS = 30      # pass-forward plays a player needs for a stable average
MIN_ROUTE_PLAYS = 10

con = duckdb.connect()
con.execute("""
CREATE VIEW plays AS
SELECT pp.nfl_id, pl.nfl_position, pl.draft_year, pp.route_ran, pp.team_coverage_man_zone AS cov,
       pp.separation_at_pass_forward AS sep, pp.cushion, pp.target::INT AS target
FROM 'work/player_play.parquet' pp
JOIN 'work/players.parquet' pl USING (nfl_id)
WHERE pp.route_ran IS NOT NULL AND pp.play_nullified_by_penalty = 'N'
""")
con.execute("""
CREATE VIEW plays_x AS
SELECT p.*,
       p.sep - avg(p.sep) OVER (PARTITION BY p.route_ran, p.cov) AS sep_over_exp
FROM plays p
""")

per_route = con.execute(f"""
SELECT nfl_id, nfl_position, draft_year, route_ran,
       count(*)                      AS routes,
       count(sep)                    AS pass_fwd,
       avg(sep)                      AS mean_sep,
       avg(sep_over_exp)             AS sep_over_exp,
       avg(cushion)                  AS mean_cushion,
       sum(target)                   AS targets
FROM plays_x
GROUP BY ALL
HAVING count(sep) >= {MIN_ROUTE_PLAYS}
""").df()
per_route.to_parquet(HERE / "work" / "wr_route_outcomes.parquet")

per_player = con.execute(f"""
SELECT nfl_id, nfl_position, draft_year,
       count(*)                      AS routes,
       count(sep)                    AS pass_fwd,
       avg(sep)                      AS mean_sep,
       avg(sep_over_exp)             AS sep_over_exp,
       avg(cushion)                  AS mean_cushion,
       sum(target)::DOUBLE / count(*) AS target_rate,
       count(DISTINCT route_ran)     AS route_types
FROM plays_x
GROUP BY ALL
HAVING count(sep) >= {MIN_PLAYS}
""").df()
per_player.to_parquet(HERE / "work" / "wr_player_outcomes.parquet")

print("players with >=", MIN_PLAYS, "pass-forward routes:", len(per_player))
print(per_player.groupby("nfl_position").size().to_string())
print("player-route rows with >=", MIN_ROUTE_PLAYS, ":", len(per_route))
print(per_route.groupby("route_ran").size().sort_values(ascending=False).to_string())
print(per_player[["pass_fwd", "mean_sep", "sep_over_exp", "target_rate"]].describe().round(2).to_string())
