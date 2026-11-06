import numpy as np
import pandas as pd
import pytest

from sbrisk.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def selections():
    rng = np.random.default_rng(1)
    rows = []
    for m in range(120):
        ko = pd.Timestamp("2021-08-14 15:00") + pd.Timedelta(days=3 * m)
        p1 = rng.dirichlet([4, 2.5, 3])
        p2 = np.array([0.55, 0.45])
        for market, sels, p in [("1X2", "ADH", p1[[2, 1, 0]]), ("OU25", "OU", p2)]:
            win = rng.choice(len(sels), p=p)
            for i, s in enumerate(sels):
                # opening price sometimes stale (generous), closing price 5% margin
                rows.append({"match_id": m, "season": "2122" if m < 60 else "2223", "kickoff": ko,
                             "market": market, "selection": s, "p_fair": p[i], "won": i == win,
                             "book_open": 1 / (p[i] * rng.uniform(0.92, 1.08)), "book_close": 1 / (p[i] * 1.05)})
    sel = pd.DataFrame(rows).sort_values(["match_id", "market", "selection"]).reset_index(drop=True)
    sel["sel_id"] = np.arange(len(sel))
    return sel


SMALL = SimConfig(n_recreational=200, organic_bets_per_match=30, n_sharps=4, n_syndicates=2,
                  n_tipsters=1, tipster_followers=(5, 5))


def test_same_seed_same_bets(selections):
    _, a = simulate(selections, SMALL)
    _, b = simulate(selections, SMALL)
    pd.testing.assert_frame_equal(a, b)


def test_labels_cover_sharps_and_syndicates_only(selections):
    accounts, _ = simulate(selections, SMALL)
    labelled = accounts.loc[accounts["label"] == 1, "group"]
    assert set(labelled) == {"sharp", "syndicate"}
    assert (accounts["group"] == "sharp").sum() == 4


def test_syndicate_members_bet_within_the_window(selections):
    accounts, bets = simulate(selections, SMALL)
    syn = bets[bets["source"] == "syndicate"].merge(accounts, on="account_id")
    spread_min = syn.groupby(["team", "sel_id"])["placed_h"].agg(lambda t: (t.max() - t.min()) * 60)
    assert (spread_min <= SMALL.syndicate_window_min + 1e-9).all()


def test_only_planted_accounts_place_informed_bets(selections):
    accounts, bets = simulate(selections, SMALL)
    informed = bets[bets["source"].isin(["sharp", "syndicate"])]
    assert accounts.set_index("account_id").loc[informed["account_id"].unique(), "label"].eq(1).all()
    assert (informed["clv"] > -0.2).all()


def test_ggr_is_stake_minus_payout(selections):
    _, bets = simulate(selections, SMALL)
    expected = bets["stake"] - np.where(bets["won"], bets["stake"] * bets["price"], 0.0)
    assert np.allclose(bets["ggr"], expected)
