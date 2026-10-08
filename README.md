# Combine tracking to NFL performance: which movement measures hold up

Entry for the NFL Big Data Bowl 2027. The question: does movement measured by 10 Hz Combine sensor
tracking predict regular-season NFL performance beyond the stopwatch and body size?

**Short answer.** We found no such measure. What we did find: most first-step measures repeat poorly
between two attempts (correlation 0.04 to 0.39), some measures only repeat the 40-yard dash, and one
offensive-line lead did not survive checking. The defensive-back question cannot be answered from
this data. The full write-up is in [WRITEUP.md](WRITEUP.md).

## Contents

| Path | What it is |
| --- | --- |
| `WRITEUP.md` | The write-up (about 1,500 words, 4 figures, 1 table) |
| `bdb2027_notebook.ipynb` | One notebook that reproduces every number and figure |
| `*.py` | The same code as separate scripts, in run order below |
| `results/` | Text output of each script, as run |
| `figures/` | The four figures, a CSV of each figure's plotted values, and the thumbnail |

## Reproducing

The contest data is **not** included here and is not redistributed. Download it from the competition
page, then either:

- **Notebook:** open `bdb2027_notebook.ipynb` on Kaggle with the competition data attached, then Run All. On a
  local machine set the environment variable `NB_DATA_DIR` to the folder that holds `combine_tracking.csv`.
- **Scripts:** place the contest CSVs in a sibling folder named `nfl-big-data-bowl-2027` and run, in order:
  `load.py`, `features.py`, `outcomes.py`, `test_wr.py`, `test_wr_matched.py`, `trench_features.py`,
  `trench_outcomes.py`, `test_trench.py --original`, `overlap.py --original`, `reliability.py`,
  `test_trench.py`, `overlap.py`, `replication.py`, `robustness_ol.py`, `db_pairs.py`, `charts.py`.

Needs Python 3 with `duckdb`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` and `pyarrow`.
All random steps use fixed seeds; running the notebook from an empty folder reproduced every script
output exactly.

## Limits

Small samples (50 to 100 players per test), a lead chosen after the results, and definitions inferred
without a data dictionary. See the write-up for the full list.
