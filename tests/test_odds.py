import numpy as np
import pandas as pd
import pytest

from sbrisk.odds import PRICE_WINDOW_H, book_price, build_selections


def _matches():
    return pd.DataFrame({
        "match_id": [0, 1], "season": ["2122", "2122"],
        "kickoff": pd.to_datetime(["2021-08-14 15:00", "2021-08-15 16:30"]),
        "FTHG": [2, 0], "FTAG": [1, 0], "FTR": ["H", "D"],
        "B365H": [2.0, 3.0], "B365D": [3.4, 3.2], "B365A": [3.8, 2.5],
        "B365CH": [1.9, 3.1], "B365CD": [3.5, 3.3], "B365CA": [4.0, 2.4],
        "PSCH": [1.95, 3.2], "PSCD": [3.6, 3.4], "PSCA": [4.2, 2.45],
        "B365>2.5": [1.8, 2.1], "B365<2.5": [2.0, 1.75],
        "B365C>2.5": [1.85, 2.0], "B365C<2.5": [1.95, 1.8],
        "PC>2.5": [1.9, 2.05], "PC<2.5": [2.0, 1.85],
    })


def test_book_price_moves_from_open_to_close():
    assert book_price(2.0, 1.8, PRICE_WINDOW_H) == pytest.approx(2.0)
    assert book_price(2.0, 1.8, 0.0) == pytest.approx(1.8)
    mid = book_price(2.0, 1.8, PRICE_WINDOW_H / 2)
    assert 1 / mid == pytest.approx((0.5 + 1 / 1.8) / 2)  # linear in implied probability
    assert book_price(2.0, 1.8, 500.0) == pytest.approx(2.0)  # clipped to the open


def test_fair_probabilities_sum_to_one_per_market():
    sel = build_selections(_matches())
    sums = sel.groupby(["match_id", "market"])["p_fair"].sum()
    assert np.allclose(sums, 1.0)


def test_exactly_one_winner_per_market():
    sel = build_selections(_matches())
    winners = sel.groupby(["match_id", "market"])["won"].sum()
    assert (winners == 1).all()
    won = sel[sel["won"]].set_index(["match_id", "market"])["selection"]
    assert won[(0, "1X2")] == "H" and won[(1, "1X2")] == "D"
    assert won[(0, "OU25")] == "O" and won[(1, "OU25")] == "U"
