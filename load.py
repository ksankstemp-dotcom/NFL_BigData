"""Convert the contest CSVs to Parquet once, so later steps query them quickly.

Run: python load.py
Reads ../nfl-big-data-bowl-2027/*.csv, writes work/*.parquet.
"""
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
SRC = HERE.parent / "nfl-big-data-bowl-2027"
OUT = HERE / "work"
OUT.mkdir(exist_ok=True)

FILES = [
    "combine_results", "combine_tracking", "players", "player_career_successes",
    "games", "player_play", "game_tracking_2023", "game_tracking_2024", "game_tracking_2025",
]

con = duckdb.connect()
for name in FILES:
    src = SRC / f"{name}.csv"
    dst = OUT / f"{name}.parquet"
    con.execute(
        f"COPY (SELECT * FROM read_csv('{src.as_posix()}', header=true, nullstr='NA', sample_size=-1)) "
        f"TO '{dst.as_posix()}' (FORMAT parquet)"
    )
    n = con.execute(f"SELECT count(*) FROM '{dst.as_posix()}'").fetchone()[0]
    print(f"{name}: {n} rows")
