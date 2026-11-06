"""Account risk scores and the trader review queue.

Four rankings, from naive to supervised:
    profit        account P&L (what a results-only screen sees)
    clv           account CLV z-score
    clv_graph     max(account CLV z, CLV z of the account's coordination group);
                  no labels needed
    supervised    gradient boosting on the features below, scored out-of-fold.
                  It needs labelled cases, which in practice come from trader
                  reviews; here they come from the simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from sbrisk.features import account_features
from sbrisk.graph import coordination_groups

MODEL_FEATURES = [
    "n_bets", "turnover", "profit", "roi", "mean_clv", "clv_z", "early_frac",
    "stake_median", "stake_cv", "median_hours_before", "in_group", "group_size",
    "group_clv", "group_clv_z",
]
METHODS = ["profit", "clv", "clv_graph", "supervised"]
FLAG_Z = 2.0  # CLV z-score at which an account or group is worth a trader's review


def build_features(bets: pd.DataFrame, account_ids, seed: int = 0) -> pd.DataFrame:
    f = account_features(bets, account_ids)
    _, groups = coordination_groups(bets, f["n_bets"], seed=seed)
    f = f.join(groups.set_index("account_id"))
    f["in_group"] = f["group_id"].notna().astype(int)
    f[["group_size", "group_clv", "group_clv_z"]] = f[["group_size", "group_clv", "group_clv_z"]].fillna(0.0)
    return f


def supervised_scores(features: pd.DataFrame, labels: pd.Series, seed: int = 0) -> pd.Series:
    """Out-of-fold probabilities, so no account is scored by a model trained on it."""
    x = features[MODEL_FEATURES].to_numpy()
    y = labels.reindex(features.index).to_numpy()
    out = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        model.fit(x[train], y[train])
        out[test] = model.predict_proba(x[test])[:, 1]
    return pd.Series(out, index=features.index)


def risk_scores(features: pd.DataFrame, labels: pd.Series | None = None, seed: int = 0) -> pd.DataFrame:
    """One column per method; higher = review first."""
    scores = pd.DataFrame(index=features.index)
    scores["profit"] = features["profit"]
    scores["clv"] = features["clv_z"]
    group_z = features["group_clv_z"].where(features["in_group"] == 1, -np.inf)
    scores["clv_graph"] = np.maximum(features["clv_z"], group_z)
    if labels is not None:
        scores["supervised"] = supervised_scores(features, labels, seed)
    return scores


def review_queue(features: pd.DataFrame, scores: pd.Series, k: int = 50) -> pd.DataFrame:
    """Top-k accounts with the evidence a trader would look at first."""
    cols = ["n_bets", "turnover", "profit", "mean_clv", "clv_z", "early_frac",
            "group_id", "group_size", "group_clv_z"]
    top = scores.sort_values(ascending=False).head(k).index
    q = features.loc[top, cols].copy()
    q.insert(0, "risk_score", scores.loc[top])
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    q["reason"] = np.select(
        [(q["group_size"] >= 2) & (q["group_clv_z"] >= FLAG_Z) & (q["group_clv_z"] >= q["clv_z"]),
         q["clv_z"] >= FLAG_Z],
        ["coordinated group beating the close", "account beating the close"],
        default="weak signal (below z = 2)")
    return q
