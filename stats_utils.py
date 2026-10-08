"""Partial rank correlation with a bootstrap interval, and Benjamini-Hochberg adjustment."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata


def _residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, v, rcond=None)
    return v - Z1 @ beta


def partial_rank_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> float:
    """Correlation of the ranks of x and y after removing the ranks of the controls Z."""
    rx, ry = rankdata(x), rankdata(y)
    if Z is not None and Z.shape[1]:
        Zr = np.column_stack([rankdata(Z[:, i]) for i in range(Z.shape[1])])
        rx, ry = _residualize(rx, Zr), _residualize(ry, Zr)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan
    return pearsonr(rx, ry)[0]


def bootstrap_partial(x, y, Z, rng, n_boot: int = 2000) -> dict:
    est = partial_rank_corr(x, y, Z)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(partial_rank_corr(x[i], y[i], None if Z is None else Z[i]))
    b = np.array(boots)
    b = b[~np.isnan(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2 * min(np.mean(b <= 0), np.mean(b >= 0))
    return {"rho": est, "ci_lo": lo, "ci_hi": hi, "p_boot": min(max(p, 1.0 / len(b)), 1.0), "boot_ok": len(b)}


def bh_adjust(p: pd.Series) -> pd.Series:
    m = len(p)
    srt = p.sort_values()
    adj = (srt * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adj.reindex(p.index)
