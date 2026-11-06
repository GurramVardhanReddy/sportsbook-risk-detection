"""Detection metrics against the planted accounts, and the GGR effect of restricting them.

Detection is scored on one period; restrictions are applied to the accounts
flagged there and measured on the next period, so the GGR figure is
out-of-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    top = scores.sort_values(ascending=False, kind="stable").head(k).index
    return float(labels.reindex(top).mean())


def recall_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    top = scores.sort_values(ascending=False, kind="stable").head(k).index
    return float(labels.reindex(top).sum() / labels.sum())


def restriction_impact(bets_next: pd.DataFrame, flagged, labels: pd.Series,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """GGR change from refusing every bet the flagged accounts place in the next period.

    Realised uses actual results (bootstrap CI over matches); expected uses the
    fair closing probabilities, so it is free of result variance.
    """
    flagged = pd.Index(flagged)
    hit = bets_next["account_id"].isin(flagged)
    total = float(bets_next["ggr"].sum())
    removed = bets_next[hit]
    is_pos = removed["account_id"].map(labels).fillna(0).astype(bool)

    per_match = (-removed.groupby("match_id")["ggr"].sum()).reindex(
        bets_next["match_id"].unique(), fill_value=0.0).to_numpy()
    rng = np.random.default_rng(seed)
    boot = per_match[rng.integers(0, len(per_match), (n_boot, len(per_match)))].sum(axis=1)

    return {
        "book_ggr": total,
        "delta_ggr": float(-removed["ggr"].sum()),
        "delta_ggr_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "delta_ggr_pct": float(-removed["ggr"].sum() / total * 100),
        "delta_expected_ggr": float(-removed["expected_ggr"].sum()),
        "from_true_positives": float(-removed.loc[is_pos, "ggr"].sum()),
        "from_false_positives": float(-removed.loc[~is_pos, "ggr"].sum()),
        "turnover_refused": float(removed["stake"].sum()),
        "turnover_refused_pct": float(removed["stake"].sum() / bets_next["stake"].sum() * 100),
    }
