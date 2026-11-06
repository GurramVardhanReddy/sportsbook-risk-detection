"""Synthetic bet flow on real odds.

Every account and bet here is SYNTHETIC. Prices and results are real (see
odds.py), so whether a bet beat the closing line comes from real line
movement, not from the simulator.

Account groups (label = 1 for the two the book wants to find):
    recreational  bet close to kickoff, favour favourites, no information
    follower      recreational accounts that copy a tipster's picks within
                  minutes of the tip: coordinated, but no edge (a decoy)
    sharp         estimate the fair closing probability with some noise and
                  bet early when the book's price beats it              label 1
    syndicate     3-8 accounts acting on one shared estimate, placing the
                  same selection within a few minutes of each other     label 1

Sharps and syndicate members also place ordinary recreational-style bets,
which dilute their individual record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sbrisk.odds import book_price

LABELLED_GROUPS = ("sharp", "syndicate")


@dataclass
class SimConfig:
    seed: int = 0
    n_recreational: int = 4000
    organic_bets_per_match: float = 250.0
    rec_hours_mean: float = 8.0          # recreational bets: hours before kickoff ~ Exp(mean)
    favourite_bias: float = 1.5          # selection weight ∝ p_fair ** bias
    n_sharps: int = 40
    sharp_match_rate: float = 0.30
    n_syndicates: int = 10
    syndicate_size: tuple[int, int] = (3, 8)
    syndicate_match_rate: float = 0.30
    # Logit-scale sd of an informed bettor's error vs the fair close, drawn per
    # sharp / syndicate. Real opening prices hold most of the edge (46% of matches
    # have a selection >2% above the fair close 96h out, 5% at 24h), so informed
    # bettors act early. With these settings sharps average about +2% CLV.
    noise_range: tuple[float, float] = (0.02, 0.12)
    min_edge: float = 0.02
    informed_hours: tuple[float, float] = (36.0, 96.0)
    participation: float = 0.75
    syndicate_window_min: float = 5.0
    n_tipsters: int = 5
    tipster_followers: tuple[int, int] = (30, 60)
    tip_rate: float = 0.25
    tip_hours: tuple[float, float] = (3.0, 30.0)
    follow_prob: float = 0.5
    follow_delay_min: float = 20.0
    informed_organic_rate: float = 0.5  # sharps' and members' recreational-style activity vs a typical account


def _round_stake(stake: np.ndarray) -> np.ndarray:
    small = np.maximum(np.round(stake), 1.0)
    return np.where(stake < 20, small, np.round(stake / 5.0) * 5.0)


def _match_grid(selections: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_matches, 5) arrays of sel_id and p_fair: 1X2 A/D/H, then OU25 O/U."""
    sel = selections.sort_values(["match_id", "market", "selection"])
    n = sel["match_id"].nunique()
    return sel["sel_id"].to_numpy().reshape(n, 5), sel["p_fair"].to_numpy().reshape(n, 5)


def _pick_recreational(rng, p_grid: np.ndarray, match_idx: np.ndarray, bias: float) -> np.ndarray:
    """Column (0-4) a recreational-style bettor picks: 1X2 70% of the time, favourite-biased."""
    w = p_grid[match_idx] ** bias
    in_1x2 = rng.random(len(match_idx)) < 0.7
    w[:, 3:] *= ~in_1x2[:, None]
    w[:, :3] *= in_1x2[:, None]
    cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
    return np.minimum((rng.random(len(match_idx))[:, None] > cum).sum(axis=1), 4)


def _informed_picks(rng, sel_grid, p_grid, open_grid, close_grid, rate, noise, cfg):
    """Matches an informed bettor acts on: (match_idx, column, hours_before)."""
    n_matches = len(sel_grid)
    m = np.flatnonzero(rng.random(n_matches) < rate)
    hours = rng.uniform(*cfg.informed_hours, size=len(m))
    price = book_price(open_grid[m], close_grid[m], hours[:, None])
    p_hat = expit(logit(p_grid[m]) + rng.normal(0.0, noise, size=price.shape))
    edge = price * p_hat - 1.0
    col = edge.argmax(axis=1)
    take = edge[np.arange(len(m)), col] > cfg.min_edge
    return m[take], col[take], hours[take]


def simulate(selections: pd.DataFrame, cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accounts, bets). Bets carry their true source only for diagnostics."""
    rng = np.random.default_rng(cfg.seed)
    sel_grid, p_grid = _match_grid(selections)
    by_id = selections.set_index("sel_id").sort_index()
    open_grid = by_id["book_open"].to_numpy()[sel_grid]
    close_grid = by_id["book_close"].to_numpy()[sel_grid]
    n_matches = len(sel_grid)

    # --- accounts ---------------------------------------------------------
    groups, teams = [], []
    groups += ["recreational"] * cfg.n_recreational
    teams += [-1] * cfg.n_recreational
    follower_sizes = rng.integers(cfg.tipster_followers[0], cfg.tipster_followers[1] + 1, cfg.n_tipsters)
    for t, size in enumerate(follower_sizes):
        groups += ["follower"] * size
        teams += [t] * size
    groups += ["sharp"] * cfg.n_sharps
    teams += [-1] * cfg.n_sharps
    syn_sizes = rng.integers(cfg.syndicate_size[0], cfg.syndicate_size[1] + 1, cfg.n_syndicates)
    for s, size in enumerate(syn_sizes):
        groups += ["syndicate"] * size
        teams += [s] * size
    accounts = pd.DataFrame({"account_id": np.arange(len(groups)), "group": groups, "team": teams})
    accounts["label"] = accounts["group"].isin(LABELLED_GROUPS).astype(int)
    accounts["activity"] = rng.lognormal(0.0, 1.0, len(accounts))
    accounts.loc[accounts["label"] == 1, "activity"] *= cfg.informed_organic_rate

    parts = []

    def add(account, match_idx, col, hours, stake, source):
        parts.append(pd.DataFrame({
            "account_id": np.asarray(account), "sel_id": sel_grid[match_idx, col],
            "hours_before": np.asarray(hours, dtype=float), "stake": _round_stake(np.asarray(stake)),
            "source": source,
        }))

    # --- organic (recreational-style) bets: every account takes part --------
    n_per_match = rng.poisson(cfg.organic_bets_per_match, n_matches)
    match_idx = np.repeat(np.arange(n_matches), n_per_match)
    weights = accounts["activity"].to_numpy()
    who = rng.choice(len(accounts), size=len(match_idx), p=weights / weights.sum())
    col = _pick_recreational(rng, p_grid, match_idx, cfg.favourite_bias)
    hours = np.clip(rng.exponential(cfg.rec_hours_mean, len(match_idx)), 0.05, 96.0)
    add(who, match_idx, col, hours, rng.lognormal(np.log(10), 1.0, len(match_idx)), "organic")

    # --- tipster picks and followers (no information) ----------------------
    for t in range(cfg.n_tipsters):
        followers = accounts.index[(accounts["group"] == "follower") & (accounts["team"] == t)].to_numpy()
        tips = np.flatnonzero(rng.random(n_matches) < cfg.tip_rate)
        tip_col = _pick_recreational(rng, p_grid, tips, cfg.favourite_bias)
        tip_h = rng.uniform(*cfg.tip_hours, size=len(tips))
        follow = rng.random((len(tips), len(followers))) < cfg.follow_prob
        ti, fi = np.nonzero(follow)
        delay_h = rng.exponential(cfg.follow_delay_min, len(ti)) / 60.0
        add(followers[fi], tips[ti], tip_col[ti], np.maximum(tip_h[ti] - delay_h, 0.02),
            rng.lognormal(np.log(10), 1.0, len(ti)), "tip")

    # --- sharps ------------------------------------------------------------
    for acc in accounts.index[accounts["group"] == "sharp"]:
        noise = rng.uniform(*cfg.noise_range)
        m, c, h = _informed_picks(rng, sel_grid, p_grid, open_grid, close_grid,
                                  cfg.sharp_match_rate, noise, cfg)
        add(np.full(len(m), acc), m, c, h, rng.lognormal(np.log(40), 0.8, len(m)), "sharp")

    # --- syndicates: one shared estimate, members place it within minutes ---
    for s in range(cfg.n_syndicates):
        members = accounts.index[(accounts["group"] == "syndicate") & (accounts["team"] == s)].to_numpy()
        noise = rng.uniform(*cfg.noise_range)
        m, c, h = _informed_picks(rng, sel_grid, p_grid, open_grid, close_grid,
                                  cfg.syndicate_match_rate, noise, cfg)
        joins = rng.random((len(m), len(members))) < cfg.participation
        di, mi = np.nonzero(joins)
        lag_h = rng.uniform(0.0, cfg.syndicate_window_min, len(di)) / 60.0
        add(members[mi], m[di], c[di], np.maximum(h[di] - lag_h, 0.02),
            rng.lognormal(np.log(30), 0.5, len(di)), "syndicate")

    bets = pd.concat(parts, ignore_index=True)
    bets = bets.join(by_id[["match_id", "season", "kickoff", "book_open", "book_close", "p_fair", "won"]],
                     on="sel_id")
    bets["price"] = book_price(bets["book_open"], bets["book_close"], bets["hours_before"])
    bets = bets.drop(columns=["book_open", "book_close"])
    t0 = bets["kickoff"].min()
    bets["placed_h"] = (bets["kickoff"] - t0).dt.total_seconds() / 3600.0 - bets["hours_before"]
    bets["clv"] = bets["price"] * bets["p_fair"] - 1.0
    bets["payout"] = np.where(bets["won"], bets["stake"] * bets["price"], 0.0)
    bets["ggr"] = bets["stake"] - bets["payout"]          # book's gross gaming revenue
    bets["expected_ggr"] = bets["stake"] * (1.0 - bets["price"] * bets["p_fair"])
    bets = bets.sort_values("placed_h").reset_index(drop=True)
    return accounts.drop(columns="activity"), bets
