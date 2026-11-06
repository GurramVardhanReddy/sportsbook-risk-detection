"""Coordination graph: accounts that keep backing the same selection within minutes.

An edge joins two accounts that placed bets on the same selection within
`window_min` minutes of each other at least `min_co` times, where those
co-bets are at least `min_overlap` of the smaller account's bets. Louvain
communities on that graph are the candidate groups.

Coordination alone is not evidence of a syndicate: tipster followers also bet
together. What separates them is the closing-line value of the bets the group
places together, so each group is scored by the CLV z-score of its co-bets.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from sbrisk.features import clv_z


def co_bets(bets: pd.DataFrame, window_min: float = 10.0) -> pd.DataFrame:
    """Every pair of bets by different accounts on one selection within the window.

    Returns columns a, b (account ids, a < b), bet_a, bet_b (row labels in `bets`).
    """
    b = bets.sort_values(["sel_id", "placed_h"])
    sel = b["sel_id"].to_numpy()
    t = b["placed_h"].to_numpy()
    acc = b["account_id"].to_numpy()
    idx = b.index.to_numpy()
    w = window_min / 60.0
    out = []
    k = 1
    while k < len(b):
        # sorted within a selection, so if no pair is close at offset k none is at k + 1
        ok = (sel[k:] == sel[:-k]) & (t[k:] - t[:-k] <= w)
        if not ok.any():
            break
        i = np.flatnonzero(ok)
        out.append((acc[i], acc[i + k], idx[i], idx[i + k]))
        k += 1
    if not out:
        return pd.DataFrame(columns=["a", "b", "bet_a", "bet_b"])
    a1, a2, b1, b2 = (np.concatenate(x) for x in zip(*out))
    keep = a1 != a2
    lo, hi = np.minimum(a1, a2)[keep], np.maximum(a1, a2)[keep]
    return pd.DataFrame({"a": lo, "b": hi, "bet_a": b1[keep], "bet_b": b2[keep]})


def coordination_groups(bets: pd.DataFrame, n_bets: pd.Series, window_min: float = 10.0,
                        min_co: int = 5, min_overlap: float = 0.10, seed: int = 0):
    """Return (edges, groups).

    edges:  a, b, co, overlap for the edges kept.
    groups: one row per account in a group of 2+: account_id, group_id, group_size,
            group_co_bets, group_clv, group_clv_z (CLV z-score of the group's co-bets).
    """
    pairs = co_bets(bets, window_min)
    edges = pairs.groupby(["a", "b"]).size().rename("co").reset_index()
    smaller = np.minimum(n_bets.reindex(edges["a"]).to_numpy(), n_bets.reindex(edges["b"]).to_numpy())
    edges["overlap"] = edges["co"] / np.maximum(smaller, 1)
    edges = edges[(edges["co"] >= min_co) & (edges["overlap"] >= min_overlap)].reset_index(drop=True)

    cols = ["account_id", "group_id", "group_size", "group_co_bets", "group_clv", "group_clv_z"]
    if edges.empty:
        return edges, pd.DataFrame(columns=cols)
    g = nx.Graph()
    g.add_weighted_edges_from(edges[["a", "b", "co"]].itertuples(index=False, name=None))
    communities = [c for c in nx.community.louvain_communities(g, weight="weight", seed=seed) if len(c) >= 2]

    member = {acc: gid for gid, comm in enumerate(communities) for acc in comm}
    pairs = pairs.assign(ga=pairs["a"].map(member), gb=pairs["b"].map(member))
    inside = pairs[pairs["ga"].notna() & (pairs["ga"] == pairs["gb"])]
    # each bet counted once per group, however many co-bets it is part of
    group_bets = pd.concat([
        inside[["ga", "bet_a"]].set_axis(["group_id", "bet"], axis=1),
        inside[["ga", "bet_b"]].set_axis(["group_id", "bet"], axis=1),
    ]).drop_duplicates()
    clv = bets.loc[group_bets["bet"], "clv"].to_numpy()
    stats = (group_bets.assign(clv=clv, clv_sq=clv**2).groupby("group_id")
             .agg(n=("clv", "size"), clv_sum=("clv", "sum"), clv_sq_sum=("clv_sq", "sum")))
    stats["group_clv"] = stats["clv_sum"] / stats["n"]
    stats["group_clv_z"] = clv_z(stats["clv_sum"], stats["clv_sq_sum"], stats["n"])

    rows = [(acc, gid, len(communities[gid])) for acc, gid in member.items()]
    groups = pd.DataFrame(rows, columns=["account_id", "group_id", "group_size"])
    groups = groups.join(stats[["n", "group_clv", "group_clv_z"]].rename(columns={"n": "group_co_bets"}),
                         on="group_id")
    groups[["group_co_bets", "group_clv", "group_clv_z"]] = (
        groups[["group_co_bets", "group_clv", "group_clv_z"]].fillna(0.0))
    return edges, groups[cols]
