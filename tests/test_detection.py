import numpy as np
import pandas as pd
import pytest

from sbrisk.detect import FLAG_Z, review_queue, risk_scores
from sbrisk.evaluate import precision_at_k, recall_at_k, restriction_impact
from sbrisk.features import MIN_BETS_FOR_Z, account_features, clv_z
from sbrisk.graph import co_bets, coordination_groups


def _bets(rows):
    b = pd.DataFrame(rows, columns=["account_id", "sel_id", "placed_h", "clv"])
    b["hours_before"] = 48.0
    b["stake"] = 10.0
    b["ggr"] = 1.0
    b["match_id"] = b["sel_id"]
    return b


def test_co_bets_pairs_only_within_window_and_across_accounts():
    b = _bets([
        (1, 7, 10.00, 0.0), (2, 7, 10.05, 0.0),   # 3 minutes apart: a pair
        (3, 7, 11.00, 0.0),                      # an hour later: no pair
        (1, 8, 10.00, 0.0), (1, 8, 10.01, 0.0),   # same account: not a pair
    ])
    pairs = co_bets(b, window_min=10)
    assert list(zip(pairs["a"], pairs["b"])) == [(1, 2)]


def test_clv_z_is_zero_below_minimum_sample():
    n = pd.Series([MIN_BETS_FOR_Z - 1, 100])
    z = clv_z(pd.Series([1.0, 5.0]), pd.Series([1.0, 1.0]), n)
    assert z.iloc[0] == 0.0 and z.iloc[1] > 0


def test_group_found_and_scored_by_its_co_bet_clv():
    rows = []
    for s in range(30):  # three accounts back the same 30 selections within a minute, +5% CLV
        for acc, lag in [(1, 0.0), (2, 0.01), (3, 0.02)]:
            rows.append((acc, s, 100.0 * s + lag, 0.05 + 0.01 * (s % 3 - 1)))
    for s in range(30):  # an unrelated account bets alone
        rows.append((4, 100 + s, 100.0 * s, -0.05))
    b = _bets(rows)
    f = account_features(b, [1, 2, 3, 4])
    edges, groups = coordination_groups(b, f["n_bets"])
    assert set(groups["account_id"]) == {1, 2, 3}
    assert groups["group_clv"].iloc[0] == pytest.approx(0.05)
    assert groups["group_clv_z"].iloc[0] > FLAG_Z


def test_graph_score_lifts_a_member_with_a_weak_own_record():
    f = pd.DataFrame({"profit": [0.0, 0.0], "clv_z": [-1.0, 3.0], "in_group": [1, 0],
                      "group_clv_z": [5.0, 0.0]}, index=pd.Index([10, 11], name="account_id"))
    s = risk_scores(f)
    assert s.loc[10, "clv_graph"] == 5.0 and s.loc[10, "clv"] == -1.0
    assert s.loc[11, "clv_graph"] == 3.0


def test_precision_and_recall_at_k():
    scores = pd.Series([0.9, 0.8, 0.7, 0.1], index=[1, 2, 3, 4])
    labels = pd.Series([1, 0, 1, 1], index=[1, 2, 3, 4])
    assert precision_at_k(scores, labels, 2) == 0.5
    assert recall_at_k(scores, labels, 3) == pytest.approx(2 / 3)


def test_restriction_impact_removes_flagged_accounts_bets():
    b = pd.DataFrame({"account_id": [1, 1, 2, 3], "match_id": [0, 1, 0, 1],
                      "ggr": [-50.0, 10.0, 20.0, 5.0], "expected_ggr": [-2.0, -1.0, 1.0, 0.5],
                      "stake": [100.0, 50.0, 40.0, 10.0]})
    labels = pd.Series({1: 1, 2: 0, 3: 0})
    out = restriction_impact(b, [1, 2], labels, n_boot=200)
    assert out["delta_ggr"] == pytest.approx(20.0)          # -( -50 + 10 + 20 )
    assert out["from_true_positives"] == pytest.approx(40.0)
    assert out["from_false_positives"] == pytest.approx(-20.0)
    assert out["delta_expected_ggr"] == pytest.approx(2.0)
    assert out["book_ggr"] == pytest.approx(-15.0)


def test_review_queue_labels_only_clear_evidence():
    f = pd.DataFrame({
        "n_bets": [50, 50, 50], "turnover": [1, 1, 1], "profit": [0, 0, 0], "mean_clv": [0, 0, 0],
        "clv_z": [0.5, 3.0, 0.1], "early_frac": [0, 0, 0], "group_id": [1.0, np.nan, 2.0],
        "group_size": [4, 0, 3], "group_clv_z": [6.0, 0.0, 0.5],
    }, index=pd.Index([1, 2, 3], name="account_id"))
    q = review_queue(f, pd.Series([6.0, 3.0, 0.5], index=[1, 2, 3]), k=3)
    assert list(q["reason"]) == ["coordinated group beating the close",
                                 "account beating the close", "weak signal (below z = 2)"]
