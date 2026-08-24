"""Phase 5 - the single evaluation on the frozen test set.

Order of operations is deliberate and enforced:

  1. verify the test-set hash written in Phase 1 still matches;
  2. load the thresholds chosen in Phase 4 on VALIDATION, verbatim;
  3. score, measure, write.

Nothing in this file selects a threshold, picks a bin count to flatter a curve,
or re-fits anything. If the numbers are bad, they are the numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import COST  # noqa: E402
from costs import block_all_cod_cost, do_nothing_cost, policy_cost  # noqa: E402
from dataset import (ARTIFACT_DIR, DATA_DIR, LABEL, LABEL_TRUE,  # noqa: E402
                     load_modelling_frame, xy)
from features import FEATURE_COLUMNS  # noqa: E402

FIG_DIR = os.path.join(ARTIFACT_DIR, "figures")
RECALL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90]


def verify_frozen_test_set(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "test_order_ids.npy")
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    recorded = None
    with open(os.path.join(data_dir, "test_set.sha256")) as f:
        for line in f:
            if line.startswith("test_order_ids.npy"):
                recorded = line.strip().split("sha256=")[1]
    if recorded is None:
        raise RuntimeError("test_set.sha256 does not record an id-file hash")
    if got != recorded:
        raise RuntimeError("FROZEN TEST SET CHANGED.\n  recorded %s\n  actual   %s"
                           % (recorded, got))
    return got


def precision_at_recall(y, p, levels=RECALL_LEVELS):
    prec, rec, thr = precision_recall_curve(y, p)
    out = {}
    for lv in levels:
        ok = rec >= lv
        if not ok.any():
            out["p_at_r%d" % int(lv * 100)] = None
            continue
        i = int(np.flatnonzero(ok)[-1])   # highest threshold still achieving this recall
        out["p_at_r%d" % int(lv * 100)] = float(prec[i])
        out["thr_at_r%d" % int(lv * 100)] = float(thr[i]) if i < len(thr) else 1.0
    return out


def reliability(y, p, n_bins=12):
    """Quantile bins, so every point on the diagram carries real mass."""
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        qs = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    rows = []
    ece = 0.0
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(dict(bin=b, n=int(m.sum()), mean_pred=float(p[m].mean()),
                         observed=float(y[m].mean())))
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return pd.DataFrame(rows), float(ece)


def plot_pr(y, p, out_path, title):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.step(rec, prec, where="post", lw=2, color="#1f4e79", label="PR-AUC = %.4f" % ap)
    ax.axhline(y.mean(), ls=":", color="#888",
               label="base rate = %.4f" % y.mean())
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_reliability(rel, ece, out_path, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="perfect calibration")
    ax.plot(rel.mean_pred, rel.observed, "o-", color="#1f4e79", lw=1.8,
            label="isotonic-calibrated model")
    for _, r in rel.iterrows():
        ax.annotate("n=%d" % r.n, (r.mean_pred, r.observed), fontsize=6,
                    xytext=(3, -8), textcoords="offset points", color="#555")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("mean predicted P(abuse)")
    ax.set_ylabel("observed abuse rate")
    ax.set_title("%s\nECE = %.5f" % (title, ece))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def band_of(p, lo, hi):
    return np.where(p >= hi, "auto_action", np.where(p >= lo, "review", "pass"))


def ring_type_table(df, p, lo, hi, label=LABEL):
    """Recall broken down by ring signature. A blended number hides this spread."""
    b = band_of(p, lo, hi)
    rows = []
    pos = df[label].to_numpy() == 1
    for rt in sorted(df.loc[pos, "ring_type"].unique()):
        m = pos & (df.ring_type.to_numpy() == rt)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append(dict(
            ring_type=rt, abuse_orders=n,
            recall_auto_action=float((b[m] == "auto_action").mean()),
            sent_to_review=float((b[m] == "review").mean()),
            missed=float((b[m] == "pass").mean()),
            recall_incl_review=float((b[m] != "pass").mean()),
            median_score=float(np.median(p[m])),
        ))
    return pd.DataFrame(rows).sort_values("recall_auto_action", ascending=False)


def hard_negative_table(df, p, lo, hi, label=LABEL):
    """Where the false positives land. Which hard negative is actually hard?"""
    b = band_of(p, lo, hi)
    neg = df[label].to_numpy() == 0
    rows = []
    for hn in sorted(df.loc[neg, "hn_type"].fillna("none").unique()):
        m = neg & (df.hn_type.fillna("none").to_numpy() == hn)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append(dict(
            population=hn, legit_cod_orders=n,
            false_positive_rate=float((b[m] == "auto_action").mean()),
            review_rate=float((b[m] == "review").mean()),
            n_false_positives=int((b[m] == "auto_action").sum()),
            median_score=float(np.median(p[m])),
        ))
    return pd.DataFrame(rows).sort_values("false_positive_rate", ascending=False)


def evaluate_one(df, bundle, lo, hi, tag, label=LABEL, make_figs=True):
    X, y = xy(df, label, cols=bundle["features"])
    p = bundle["calibrator"].predict_proba(X)[:, 1]
    v = df["value_inr"].to_numpy()

    pol = policy_cost(y, p, v, lo, hi)
    single = policy_cost(y, p, v, hi, hi)
    nothing = do_nothing_cost(y, v)
    blockall = block_all_cod_cost(y, v)
    rel, ece = reliability(y, p)

    tp, fp, fn = pol["tp"], pol["fp"], pol["fn"]
    prec_action = tp / (tp + fp) if (tp + fp) else 0.0
    rec_action = tp / max(int(y.sum()), 1)

    m = dict(
        tag=tag, n_orders=int(len(y)), n_abuse=int(y.sum()),
        base_rate=float(y.mean()),
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=ece,
        band_lo=lo, band_hi=hi,
        precision_auto_action=float(prec_action),
        recall_auto_action=float(rec_action),
        recall_incl_review=float((tp + pol["caught_in_review"]) / max(int(y.sum()), 1)),
        pass_share=pol["pass_share"], review_share=pol["review_share"],
        action_share=pol["action_share"],
        n_review=pol["n_review"], tp=tp, fp=fp, fn=fn, tn=pol["tn"],
        cost_per_1000_policy=pol["cost_per_1000"],
        cost_per_1000_single_threshold=single["cost_per_1000"],
        cost_per_1000_do_nothing=nothing["cost_per_1000"],
        cost_per_1000_block_all_cod=blockall["cost_per_1000"],
    )
    m["saving_vs_do_nothing_per_1000"] = (m["cost_per_1000_do_nothing"]
                                          - m["cost_per_1000_policy"])
    m["saving_vs_do_nothing_pct"] = (100.0 * m["saving_vs_do_nothing_per_1000"]
                                     / m["cost_per_1000_do_nothing"])
    m.update(precision_at_recall(y, p))

    if make_figs:
        plot_pr(y, p, os.path.join(FIG_DIR, "pr_curve_%s.png" % tag),
                "Precision-recall, %s (frozen test set)" % tag)
        plot_reliability(rel, ece, os.path.join(FIG_DIR, "reliability_%s.png" % tag),
                         "Reliability, %s" % tag)
        rel.to_csv(os.path.join(ARTIFACT_DIR, "reliability_%s.csv" % tag), index=False)
    return m, p


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    h = verify_frozen_test_set()
    print("[eval] frozen test set verified, sha256=%s" % h[:16])

    with open(os.path.join(ARTIFACT_DIR, "policy.json")) as f:
        policies = json.load(f)
    lo, hi = policies["ring_grouped"]["band_lo"], policies["ring_grouped"]["band_hi"]
    lo_t, hi_t = policies["temporal"]["band_lo"], policies["temporal"]["band_hi"]
    print("[eval] validation-selected bands: ring_grouped lo=%.4f hi=%.4f | "
          "temporal lo=%.4f hi=%.4f (never re-tuned on test)" % (lo, hi, lo_t, hi_t))

    df = load_modelling_frame()
    with open(os.path.join(ARTIFACT_DIR, "model_primary.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "model_temporal.pkl"), "rb") as f:
        bundle_t = pickle.load(f)

    results = {}

    # ---- primary: ring-grouped held-out test -------------------------------
    te = df.loc[df.split == "test"].reset_index(drop=True)
    m, p = evaluate_one(te, bundle, lo, hi, "ring_grouped")
    results["ring_grouped"] = m
    print("[eval] ring-grouped:", json.dumps({k: m[k] for k in
          ("n_orders", "n_abuse", "base_rate", "pr_auc", "precision_auto_action",
           "recall_auto_action", "cost_per_1000_policy")}, indent=2))

    rt = ring_type_table(te, p, lo, hi)
    rt.to_csv(os.path.join(ARTIFACT_DIR, "recall_by_ring_type.csv"), index=False)
    print(rt.to_string(index=False))

    hn = hard_negative_table(te, p, lo, hi)
    hn.to_csv(os.path.join(ARTIFACT_DIR, "false_positives_by_population.csv"), index=False)
    print(hn.to_string(index=False))

    # ---- ceiling check: same model, noiseless labels -----------------------
    m_true, _ = evaluate_one(te, bundle, lo, hi, "ring_grouped_truelabels",
                             label=LABEL_TRUE, make_figs=False)
    results["ring_grouped_vs_true_labels"] = m_true

    # ---- secondary: temporal holdout ---------------------------------------
    tt = df.loc[df.temporal_split == "test"].reset_index(drop=True)
    m_t, p_t = evaluate_one(tt, bundle_t, lo_t, hi_t, "temporal")
    results["temporal"] = m_t
    rt_t = ring_type_table(tt, p_t, lo_t, hi_t)
    rt_t.to_csv(os.path.join(ARTIFACT_DIR, "recall_by_ring_type_temporal.csv"), index=False)
    print("[eval] temporal:", json.dumps({k: m_t[k] for k in
          ("n_orders", "n_abuse", "base_rate", "pr_auc", "precision_auto_action",
           "recall_auto_action", "cost_per_1000_policy")}, indent=2))
    print(rt_t.to_string(index=False))

    results["policy"] = policies
    results["frozen_test_sha256"] = h
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- scored test set + attribution payloads for the UI -----------------
    te_out = te[["order_id", "customer_id", "order_ts", "value_inr", "category",
                 "ring_type", "hn_type", LABEL, LABEL_TRUE]].copy()
    te_out["p"] = p
    te_out["band"] = band_of(p, lo, hi)
    te_out.to_parquet(os.path.join(ARTIFACT_DIR, "scored_test.parquet"), index=False)

    from explain import Attributor, build_payload, write_explanation
    att = Attributor(bundle)
    flagged = np.flatnonzero(te_out.band.to_numpy() != "pass")
    print("[eval] computing TreeSHAP for %d flagged orders" % len(flagged))
    Xf = te[bundle["features"]].to_numpy(dtype=np.float32)[flagged]
    sv = att.shap_values(Xf)
    payloads = []
    for k, i in enumerate(flagged):
        facts = dict(order_id=int(te_out.order_id.iloc[i]),
                     order_ts=str(te_out.order_ts.iloc[i]),
                     value_inr=float(te_out.value_inr.iloc[i]),
                     category=str(te_out.category.iloc[i]),
                     payment_mode="COD")
        payloads.append(build_payload(facts, Xf[k], sv[k], att,
                                      float(te_out.p.iloc[i]), str(te_out.band.iloc[i])))
    with open(os.path.join(ARTIFACT_DIR, "explanation_payloads.json"), "w") as f:
        json.dump(payloads, f, indent=1)

    sample = payloads[:6]
    written = []
    for pl in sample:
        text, src = write_explanation(pl)
        written.append(dict(order_id=pl["order"]["order_id"], probability=pl["probability"],
                            band=pl["decision_band"], writer=src, explanation=text))
    with open(os.path.join(ARTIFACT_DIR, "explanations_sample.json"), "w") as f:
        json.dump(written, f, indent=2)
    print("[eval] wrote metrics.json, scored_test.parquet, explanation payloads (%d)"
          % len(payloads))


if __name__ == "__main__":
    main()
