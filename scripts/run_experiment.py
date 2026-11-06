"""End-to-end experiment: simulate, detect on 2021/22-2022/23, restrict on 2023/24-2024/25.

Usage:
    python scripts/run_experiment.py                 # 5 seeds
    python scripts/run_experiment.py --seeds 0 --k 100

Writes reports/results.json, reports/review_queue_seed0.csv and two figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbrisk.detect import FLAG_Z, METHODS, build_features, review_queue, risk_scores  # noqa: E402
from sbrisk.evaluate import precision_at_k, recall_at_k, restriction_impact  # noqa: E402
from sbrisk.odds import build_selections, load_matches  # noqa: E402
from sbrisk.simulate import SimConfig, simulate  # noqa: E402

DETECT_SEASONS = ["2122", "2223"]
RESTRICT_SEASONS = ["2324", "2425"]
KS = [25, 50, 100]


def run_seed(selections: pd.DataFrame, seed: int, k_restrict: int) -> dict:
    accounts, bets = simulate(selections, SimConfig(seed=seed))
    labels = accounts.set_index("account_id")["label"]
    detect = bets[bets["season"].isin(DETECT_SEASONS)]
    restrict = bets[bets["season"].isin(RESTRICT_SEASONS)]

    features = build_features(detect, accounts["account_id"], seed=seed)
    scores = risk_scores(features, labels, seed=seed)

    out = {"seed": seed, "accounts": len(accounts), "positives": int(labels.sum()),
           "bets": len(bets), "detection": {}, "restriction": {}}
    for m in METHODS:
        out["detection"][m] = {
            **{f"precision@{k}": precision_at_k(scores[m], labels, k) for k in KS},
            **{f"recall@{k}": recall_at_k(scores[m], labels, k) for k in KS},
        }
        flagged = scores[m].sort_values(ascending=False, kind="stable").head(k_restrict).index
        out["restriction"][m] = restriction_impact(restrict, flagged, labels, seed=seed)
    # a trader queue flags only clear evidence: CLV z >= FLAG_Z, not a fixed k
    out["threshold"] = {}
    for m in ["clv", "clv_graph"]:
        flagged = scores.index[scores[m] >= FLAG_Z]
        out["threshold"][m] = {"flagged": len(flagged),
                               "precision": float(labels.reindex(flagged).mean()) if len(flagged) else 0.0,
                               "recall": float(labels.reindex(flagged).sum() / labels.sum()),
                               **restriction_impact(restrict, flagged, labels, seed=seed)}
    out["restriction"]["all_planted"] = restriction_impact(
        restrict, labels.index[labels == 1], labels, seed=seed)

    # how many planted syndicates the graph recovered as a group of 2+
    syn = accounts[accounts["group"] == "syndicate"].set_index("account_id")["team"]
    grouped = features.loc[syn.index, "group_id"]
    out["syndicates_recovered"] = int(grouped.groupby(syn).apply(lambda g: g.notna().sum() >= 2).sum())
    out["syndicates_planted"] = int(syn.nunique())
    if seed == 0:
        queue = review_queue(features, scores["clv_graph"], k=k_restrict)
        queue["planted_group"] = accounts.set_index("account_id").loc[queue.index, "group"]
        queue.to_csv(ROOT / "reports" / f"review_queue_seed{seed}.csv")
    return out


def summarise(runs: list[dict]) -> dict:
    def agg(path):
        vals = np.array([path(r) for r in runs], dtype=float)
        return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

    summary = {"detection": {}, "restriction": {}}
    for m in METHODS:
        summary["detection"][m] = {key: agg(lambda r, m=m, key=key: r["detection"][m][key])
                                   for key in runs[0]["detection"][m]}
    keys = ["delta_ggr", "delta_ggr_pct", "delta_expected_ggr", "from_true_positives",
            "from_false_positives", "turnover_refused_pct"]
    for m in METHODS + ["all_planted"]:
        summary["restriction"][m] = {key: agg(lambda r, m=m, key=key: r["restriction"][m][key]) for key in keys}
    summary["threshold"] = {
        m: {key: agg(lambda r, m=m, key=key: r["threshold"][m][key])
            for key in ["flagged", "precision", "recall"] + keys}
        for m in runs[0]["threshold"]}
    summary["book_ggr"] = agg(lambda r: r["restriction"]["profit"]["book_ggr"])
    summary["syndicates_recovered"] = agg(lambda r: r["syndicates_recovered"])
    return summary


def plot(runs: list[dict], summary: dict, k_restrict: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"profit": "Profit (P&L)", "clv": "CLV z-score", "clv_graph": "CLV + coordination graph",
             "supervised": "Supervised (out-of-fold)"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in METHODS:
        means = [summary["detection"][m][f"precision@{k}"]["mean"] for k in KS]
        ax.plot(KS, means, marker="o", label=names[m])
    ax.set_xlabel("Review queue size k")
    ax.set_ylabel("Precision@k (planted sharps + syndicates)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(KS)
    ax.legend(frameon=False)
    ax.set_title(f"Detection, mean of {len(runs)} seeds (synthetic accounts, real odds)")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "precision_at_k.png", dpi=150)

    # one seed: the seeds share the same real match results, so the bootstrap
    # over matches, not the spread across seeds, is the uncertainty in realised GGR
    r0 = runs[0]["restriction"]
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = METHODS + ["all_planted"]
    labels = [names.get(m, "All planted accounts") for m in keys]
    real = np.array([r0[m]["delta_ggr"] for m in keys]) / 1000
    lo = np.array([r0[m]["delta_ggr_ci95"][0] for m in keys]) / 1000
    hi = np.array([r0[m]["delta_ggr_ci95"][1] for m in keys]) / 1000
    exp = np.array([r0[m]["delta_expected_ggr"] for m in keys]) / 1000
    x = np.arange(len(keys))
    ax.bar(x - 0.2, real, 0.4, yerr=[real - lo, hi - real], capsize=3, label="Realised (95% CI over matches)")
    ax.bar(x + 0.2, exp, 0.4, label="Expected from fair closing prices")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("GGR change, thousands")
    ax.set_title(f"Refusing the top {k_restrict} accounts in the next two seasons (seed {runs[0]['seed']})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "ggr_impact.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=100, help="accounts restricted per method")
    args = ap.parse_args()

    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    selections = build_selections(load_matches())
    runs = []
    for seed in args.seeds:
        t = time.time()
        runs.append(run_seed(selections, seed, args.k))
        r = runs[-1]
        print(f"seed {seed}: {r['bets']} bets, {r['positives']} planted, "
              f"syndicates recovered {r['syndicates_recovered']}/{r['syndicates_planted']} "
              f"({time.time() - t:.0f}s)")
        for m in METHODS:
            d, g = r["detection"][m], r["restriction"][m]
            print(f"  {m:<11} P@50 {d['precision@50']:.2f}  P@100 {d['precision@100']:.2f}  "
                  f"dGGR {g['delta_ggr']:>10,.0f} ({g['delta_ggr_pct']:+.2f}%)  "
                  f"expected {g['delta_expected_ggr']:>9,.0f}")
    summary = summarise(runs)
    result = {"config": {"detect_seasons": DETECT_SEASONS, "restrict_seasons": RESTRICT_SEASONS,
                         "k_restrict": args.k, "sim": SimConfig().__dict__},
              "summary": summary, "runs": runs}
    (ROOT / "reports" / "results.json").write_text(json.dumps(result, indent=1, default=str))
    plot(runs, summary, args.k)
    print("saved reports/results.json and reports/figures/")


if __name__ == "__main__":
    main()
