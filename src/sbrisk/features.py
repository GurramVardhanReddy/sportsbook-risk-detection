"""Per-account features from bet history. Uses no labels and no bet `source`."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BETS_FOR_Z = 5
EARLY_HOURS = 24.0


def clv_z(clv_sum: pd.Series, clv_sq_sum: pd.Series, n: pd.Series, sd_floor: float = 0.05) -> pd.Series:
    """t-statistic of mean CLV against 0. The sd floor stops tiny samples looking certain."""
    mean = clv_sum / n
    var = (clv_sq_sum / n - mean**2).clip(lower=0.0) * n / (n - 1).clip(lower=1)
    sd = np.sqrt(var).clip(lower=sd_floor)
    z = mean / (sd / np.sqrt(n))
    return z.where(n >= MIN_BETS_FOR_Z, 0.0)


def account_features(bets: pd.DataFrame, account_ids) -> pd.DataFrame:
    """One row per account (accounts with no bets get zeros)."""
    b = bets.assign(clv_sq=bets["clv"] ** 2, early=bets["hours_before"] > EARLY_HOURS,
                    profit=-bets["ggr"])
    g = b.groupby("account_id")
    f = pd.DataFrame({
        "n_bets": g.size(),
        "turnover": g["stake"].sum(),
        "profit": g["profit"].sum(),
        "clv_sum": g["clv"].sum(),
        "clv_sq_sum": g["clv_sq"].sum(),
        "early_frac": g["early"].mean(),
        "stake_median": g["stake"].median(),
        "stake_cv": g["stake"].std().fillna(0.0) / g["stake"].mean(),
        "median_hours_before": g["hours_before"].median(),
    })
    f["mean_clv"] = f["clv_sum"] / f["n_bets"]
    f["clv_z"] = clv_z(f["clv_sum"], f["clv_sq_sum"], f["n_bets"])
    f["roi"] = f["profit"] / f["turnover"]
    f = f.reindex(pd.Index(account_ids, name="account_id")).fillna(0.0)
    return f
