"""Break features for WR/DB combine drills, computed from positions.

Speed and heading are recomputed from smoothed x/y rather than taken from the s, a and
dis columns: in this data dis*10 and s differ in scale (means 3.0 vs 4.9 yd/s), so the
given columns are not trusted until that is explained.

Run: python features.py   (needs work/combine_tracking.parquet from load.py)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DT = 0.1            # seconds per frame (10 Hz)
SMOOTH = 3          # frames in the centred moving average
TURN_WINDOW = 5     # frames over which a heading change is measured
MIN_SPEED = 2.0     # yd/s; the break must happen while moving
DRILL_TYPES = ("SKILL_DRILLS_WR", "SKILL_DRILLS_DB")


def _smooth(v: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    pad = k // 2
    p = np.pad(v, pad, mode="edge")
    return np.convolve(p, np.ones(k) / k, mode="valid")


def _wrap(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def attempt_features(g: pd.DataFrame) -> dict | None:
    """One athlete, one attempt, frames in time order."""
    if len(g) < TURN_WINDOW + 6:
        return None
    x, y = _smooth(g["x"].to_numpy()), _smooth(g["y"].to_numpy())
    vx, vy = np.gradient(x, DT), np.gradient(y, DT)
    speed = np.hypot(vx, vy)
    heading = np.degrees(np.arctan2(vy, vx))
    turn = np.abs(_wrap(heading[TURN_WINDOW:] - heading[:-TURN_WINDOW]))
    moving = speed[:-TURN_WINDOW] >= MIN_SPEED
    if not moving.any():
        return None
    turn = np.where(moving, turn, 0.0)
    i = int(np.argmax(turn))             # window start; the break lies in [i, i + TURN_WINDOW]
    j = i + TURN_WINDOW
    entry = float(speed[max(0, i - 3):i + 1].mean())
    seg = speed[i:j + 1]
    min_speed = float(seg.min())
    k = i + int(seg.argmin())
    target = 0.9 * entry
    after = np.nonzero(speed[k:] >= target)[0]
    reaccel = float(after[0] * DT) if len(after) else np.nan
    return {
        "frames": len(g),
        "break_angle": float(turn[i]),
        "break_frame": i,
        "entry_speed": entry,
        "min_speed": min_speed,
        "speed_lost": entry - min_speed,
        "reaccel_s": reaccel,
        "peak_speed": float(speed.max()),
        "path_len": float(np.hypot(np.diff(x), np.diff(y)).sum()),
    }


def build(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT draft_year, event_id, nfl_id, drill_type, drill_name, attempt, time, x, y
        FROM 'work/combine_tracking.parquet'
        WHERE entity_type = 'PLAYER' AND drill_type IN ('SKILL_DRILLS_WR', 'SKILL_DRILLS_DB')
        ORDER BY event_id, nfl_id, time
        """
    ).df()
    rows = []
    for (ev, nid), g in df.groupby(["event_id", "nfl_id"], sort=False):
        f = attempt_features(g)
        if f is None:
            continue
        first = g.iloc[0]
        rows.append({"event_id": ev, "nfl_id": nid, "draft_year": first.draft_year,
                     "drill_type": first.drill_type, "drill_name": first.drill_name,
                     "attempt": first.attempt, **f})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    con = duckdb.connect()
    out = build(con)
    out.to_parquet(HERE / "work" / "route_features.parquet")
    print(len(out), "attempts")
    pd.set_option("display.width", 200)
    print(out.groupby("drill_name")[["break_angle", "entry_speed", "speed_lost", "reaccel_s"]]
          .median().round(2).to_string())
