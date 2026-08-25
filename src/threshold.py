"""Phase 4 - threshold and band selection, on validation only.

Nothing in this file is ever allowed to see the test set. The selected
thresholds are written to artifacts/policy.json and Phase 5 reads them back
verbatim.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import COST  # noqa: E402
from costs import block_all_cod_cost, do_nothing_cost, policy_cost  # noqa: E402
from dataset import ARTIFACT_DIR, LABEL  # noqa: E402

FIG_DIR = os.path.join(ARTIFACT_DIR, "figures")


def sweep_single(y, p, value, n_points=400, cost=COST):
    """Single cut-point sweep: block if p >= t, else pass. No review band."""
    ts = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_points)))
    ts = np.clip(np.concatenate([[0.0], ts, [1.0]]), 0, 1)
    rows = []
    for t in ts:
        r = policy_cost(y, p, value, t, t, cost)
        r["threshold"] = float(t)
        rows.append(r)
    return pd.DataFrame(rows)


def sweep_bands(y, p, value, cost=COST, n_lo=60, n_hi=60):
    los = np.linspace(cost.band_search_lo[0], cost.band_search_lo[1], n_lo)
    his = np.linspace(cost.band_search_hi[0], cost.band_search_hi[1], n_hi)
    rows = []
    for lo in los:
        for hi in his:
            if hi <= lo:
                continue
            r = policy_cost(y, p, value, lo, hi, cost)
            r["lo"] = float(lo)
            r["hi"] = float(hi)
            rows.append(r)
    return pd.DataFrame(rows)


def plot_cost_curve(curve, best_t, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(curve.threshold, curve.cost_per_1000, lw=2, color="#1f4e79")
    ax.axvline(best_t, color="#c00000", ls="--", lw=1.4,
               label="cost-minimising t = %.4f" % best_t)
    ymin = curve.cost_per_1000.min()
    ax.scatter([best_t], [ymin], color="#c00000", zorder=5)
    ax.set_xlabel("threshold on calibrated P(abuse)")
    ax.set_ylabel("expected cost per 1000 COD orders (INR)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def select_policy(va, tag, make_fig=True):
    """Choose the single threshold and the three-band policy for ONE model.

    Called once per model. The temporal model gets its own selection on its own
    validation fold: reusing the primary model's cut points would confound
    "does the detector degrade over time" with "do two models put their scores
    on slightly different scales", and only the first question is interesting.
    """
    y = va[LABEL].to_numpy()
    p = va["p"].to_numpy()
    v = va["value_inr"].to_numpy()

    nothing = do_nothing_cost(y, v)
    blockall = block_all_cod_cost(y, v)

    curve = sweep_single(y, p, v)
    best = curve.loc[curve.cost_per_1000.idxmin()]
    if make_fig:
        plot_cost_curve(curve, float(best.threshold),
                        os.path.join(FIG_DIR, "cost_curve_%s.png" % tag),
                        "Cost-based threshold selection - %s (validation fold)" % tag)
    curve.to_csv(os.path.join(ARTIFACT_DIR, "cost_curve_%s.csv" % tag), index=False)

    bands = sweep_bands(y, p, v)
    feasible = bands.loc[bands.review_share <= COST.max_review_share]
    if len(feasible) == 0:
        raise RuntimeError("no band configuration satisfies the review-capacity constraint")
    bb = feasible.loc[feasible.cost_per_1000.idxmin()]

    policy = dict(
        selected_on="validation fold of the %s split" % tag,
        single_threshold=float(best.threshold),
        single_threshold_cost_per_1000=float(best.cost_per_1000),
        band_lo=float(bb.lo), band_hi=float(bb.hi),
        band_cost_per_1000=float(bb.cost_per_1000),
        band_pass_share=float(bb.pass_share),
        band_review_share=float(bb.review_share),
        band_action_share=float(bb.action_share),
        review_capacity_constraint=COST.max_review_share,
        baseline_do_nothing_per_1000=float(nothing["cost_per_1000"]),
        baseline_block_all_cod_per_1000=float(blockall["cost_per_1000"]),
        val_n=int(len(va)), val_positives=int(y.sum()),
    )
    policy["saving_vs_do_nothing_per_1000"] = (
        policy["baseline_do_nothing_per_1000"] - policy["band_cost_per_1000"])
    policy["saving_vs_do_nothing_pct"] = (
        100.0 * policy["saving_vs_do_nothing_per_1000"]
        / policy["baseline_do_nothing_per_1000"])

    bands.to_csv(os.path.join(ARTIFACT_DIR, "band_sweep_%s.csv" % tag), index=False)
    return policy


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    out = {}
    for tag, fname in [("ring_grouped", "val_predictions_primary.parquet"),
                       ("temporal", "val_predictions_temporal.parquet")]:
        va = pd.read_parquet(os.path.join(ARTIFACT_DIR, fname))
        out[tag] = select_policy(va, tag)
        print("[thr] %s" % tag, json.dumps(out[tag], indent=2))
    with open(os.path.join(ARTIFACT_DIR, "policy.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
