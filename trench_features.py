"""Drill-general movement features for OL and DL combine drills, from positions.

Unlike the route drills there is no single break to find, so the features describe how
the athlete gets going and how much they change direction:

  start         first frame where smoothed speed reaches START_SPEED
  speed_0p5s    speed 0.5 s after start
  burst_dist_1s distance covered in the first 1.0 s after start
  peak_accel_1s highest acceleration (rate of change of speed) in the first 1.0 s
  peak_speed    highest speed in the attempt
  mean_speed    mean speed while moving
  turn_rate     total absolute heading change per second while moving (direction changes)

Run: python trench_features.py   (needs work/combine_tracking.parquet)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from features import DT, _smooth

HERE = Path(__file__).parent
START_SPEED = 1.0   # yd/s
MOVING = 1.0        # yd/s
GROUPS = {
    "OL": ("SKILL_DRILLS_OL", {
        "PASS_PRO_MIRROR": ("PASS_PRO_MIRROR_DRILL", "PASS_PRO-MIRROR_DRILL"),
        "PASS_RUSH_DROP": ("PASS_RUSH_DROP",),
        "WAVE_SLIDE": ("FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",),
    }),
    "DL": ("SKILL_DRILLS_DL", {
        "PASS_RUSH": ("PASS_RUSH_DRILL",),
        "FRONT_START_WAVE": ("FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION",),
        "FOUR_BAG": ("FOUR_BAG_AGILITY_DRILL",),
        "BODY_CONTROL": ("BODY_CONTROL_DRILL",),
    }),
}


def attempt(g: pd.DataFrame) -> dict | None:
    if len(g) < 15:
        return None
    x, y = _smooth(g["x"].to_numpy()), _smooth(g["y"].to_numpy())
    vx, vy = np.gradient(x, DT), np.gradient(y, DT)
    speed = np.hypot(vx, vy)
    acc = np.gradient(speed, DT)
    go = np.nonzero(speed >= START_SPEED)[0]
    if len(go) == 0:
        return None
    s = int(go[0])
    if s + 10 >= len(speed):
        return None
    step = np.hypot(np.diff(x), np.diff(y))
    heading = np.unwrap(np.arctan2(vy, vx))
    mv = speed[1:] >= MOVING
    dh = np.abs(np.diff(heading))[mv[: len(heading) - 1]] if mv.any() else np.array([0.0])
    # Start-free features: the best window anywhere in the attempt, so they do not depend on
    # which frame is called the start. "_s" versions use the file's own speed and acceleration.
    sg, ag = g["s"].to_numpy(float), g["a"].to_numpy(float)
    xr, yr = g["x"].to_numpy(float), g["y"].to_numpy(float)
    n = len(g)
    net10 = np.hypot(xr[10:] - xr[:-10], yr[10:] - yr[:-10]) if n > 10 else np.array([0.0])
    gain5 = speed[5:] - speed[:-5]
    gain5_s = sg[5:] - sg[:-5]
    path10_s = np.convolve(sg * DT, np.ones(10), mode="valid") if n > 10 else np.array([0.0])
    top3 = np.sort(ag)[-3:].mean()
    return {
        "frames": len(g),
        "best_net_1s": float(net10.max()),
        "best_gain_0p5s": float(gain5.max()),
        "best_gain_0p5s_s": float(gain5_s.max()),
        "best_path_1s_s": float(path10_s.max()),
        "peak_speed_s": float(sg.max()),
        "peak_accel_s": float(ag.max()),
        "top3_accel_s": float(top3),
        "speed_0p5s": float(speed[s + 5]),
        "burst_dist_1s": float(step[s:s + 10].sum()),
        "peak_accel_1s": float(acc[s:s + 10].max()),
        "peak_speed": float(speed.max()),
        "mean_speed": float(speed[speed >= MOVING].mean()),
        "turn_rate": float(np.degrees(dh.sum()) / max(mv.sum() * DT, DT)),
    }


def build() -> pd.DataFrame:
    con = duckdb.connect()
    parts = []
    for side, (dtype, drills) in GROUPS.items():
        names = [n for v in drills.values() for n in v]
        df = con.execute(
            "SELECT event_id, nfl_id, drill_name, time, x, y, s, a FROM 'work/combine_tracking.parquet' "
            "WHERE entity_type = 'PLAYER' AND drill_type = ? AND drill_name IN (SELECT unnest(?)) "
            "ORDER BY event_id, nfl_id, time", [dtype, names]
        ).df()
        key = {n: k for k, v in drills.items() for n in v}
        for (ev, nid), g in df.groupby(["event_id", "nfl_id"], sort=False):
            f = attempt(g)
            if f:
                parts.append({"side": side, "drill": key[g.drill_name.iloc[0]], "event_id": ev, "nfl_id": nid, **f})
    return pd.DataFrame(parts)


if __name__ == "__main__":
    out = build()
    out.to_parquet(HERE / "work" / "trench_features.parquet")
    print(len(out), "attempts")
    pd.set_option("display.width", 200)
    print(out.groupby(["side", "drill"]).agg(n=("nfl_id", "size"), players=("nfl_id", "nunique"),
          speed_0p5s=("speed_0p5s", "median"), burst=("burst_dist_1s", "median"),
          accel=("peak_accel_1s", "median"), peak=("peak_speed", "median"),
          turn=("turn_rate", "median")).round(2).to_string())
