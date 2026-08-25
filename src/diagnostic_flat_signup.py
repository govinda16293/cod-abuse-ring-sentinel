"""Generator diagnostic - does the signup ramp manufacture the temporal collapse?

NOT A HEADLINE RESULT. This regenerates a SECOND dataset into data_flat/ with one
change, and evaluates the temporal split on it only.

Precisely what this does to data/ (worth stating exactly rather than claiming
more than is true): it READS data/ to compute descriptive population statistics
for the ramped side of the comparison - unseen-customer share, median tenure,
monthly volume - and it reads back the already-published temporal metrics from
artifacts/metrics.json. It does NOT score the ring-grouped frozen test set, does
not fit or calibrate anything on data/, does not select any threshold from it,
and changes no published number. data/test_set.sha256 is unchanged by a run.

The claim being tested. In the headline generator every customer draws the same
expected order count regardless of signup date, so a customer who joins five days
before the window ends crams all of their orders into those five days. Two
consequences at late calendar times: order volume ramps (8.7k COD orders in month
1 to 27.6k in month 18), and the population acquires a large mass of accounts
that are simultaneously brand-new and placing several orders quickly - which is
the R4_burst_value signature.

Since 99.3% of the temporal false positives were exactly that shape, the honest
question is how much of the reported precision drop is a real drift result and
how much is my generator. `flat_signup=True` scales each customer's order rate by
the fraction of the window they were present for, making orders-per-day flat
across calendar time. Everything else - seed, ring planting, hard negatives,
label noise, feature code, model, band-selection procedure - is identical.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARTIFACT_DIR, LABEL, active_features, load_modelling_frame  # noqa: E402
from evaluate import evaluate_one, ring_type_table  # noqa: E402
from threshold import select_policy  # noqa: E402
from train import train_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAT_DIR = os.path.join(ROOT, "data_flat")
PY = sys.executable


def build_flat_dataset():
    env = dict(os.environ)
    env["SENTINEL_DATA_DIR"] = FLAT_DIR
    env["SENTINEL_FLAT_SIGNUP"] = "1"
    os.makedirs(FLAT_DIR, exist_ok=True)
    for script in ["src/generate_data.py", "src/features.py"]:
        print("[flat] running %s into data_flat/" % script, flush=True)
        r = subprocess.run([PY, os.path.join(ROOT, script)], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            raise SystemExit("failed: %s" % script)


def monthly_profile(df):
    m = df.assign(month=df.order_ts.dt.to_period("M").astype(str)).groupby("month")
    return m.agg(cod_orders=(LABEL, "size"), abuse_rate=(LABEL, "mean"))


def temporal_run(df, tag):
    """Train on months 1-12, select bands on its own val fold, evaluate 13-18."""
    dt = df.loc[df.temporal_split != "drop"]
    clf, cal, tr, va, p_va, info = train_split(dt, "temporal_split")
    vt = va[["order_id", "group_key", "ring_type", LABEL, "value_inr"]].copy()
    vt["p"] = p_va
    pol = select_policy(vt, tag, make_fig=False)
    te = df.loc[df.temporal_split == "test"].reset_index(drop=True)
    bundle = {"booster": clf, "calibrator": cal, "features": active_features()}
    m, p = evaluate_one(te, bundle, pol["band_lo"], pol["band_hi"], tag, make_figs=False)
    rt = ring_type_table(te, p, pol["band_lo"], pol["band_hi"])
    return m, rt, pol, te, p


def main():
    if not os.path.exists(os.path.join(FLAT_DIR, "features.parquet")):
        build_flat_dataset()

    flat = load_modelling_frame(FLAT_DIR)
    ramped = load_modelling_frame()          # data/, read-only, temporal split only

    print("\n[flat] === monthly profile, flat-signup dataset ===")
    prof_flat = monthly_profile(flat)
    print(prof_flat.to_string())

    m_flat, rt_flat, pol_flat, te_flat, p_flat = temporal_run(flat, "flat_temporal")

    # the ramped temporal numbers already exist from the headline run
    with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
        m_ramp = json.load(f)["temporal"]

    prof_ramp = monthly_profile(ramped)

    # how much of the late-period population is brand new, in each world?
    def new_share(df):
        tr = set(df.loc[df.temporal_split == "train", "customer_id"].unique().tolist())
        te = df.loc[(df.temporal_split == "test") & (df[LABEL] == 0)]
        return float((~te.customer_id.isin(tr)).mean()), float(te.cust_tenure_days.median())

    ns_flat, ten_flat = new_share(flat)
    ns_ramp, ten_ramp = new_share(ramped)

    keys = ["n_orders", "n_abuse", "base_rate", "pr_auc", "precision_auto_action",
            "recall_auto_action", "review_share", "cost_per_1000_policy"]
    comp = pd.DataFrame([
        dict(metric=k, ramped=m_ramp.get(k), flat_signup=m_flat.get(k)) for k in keys]
        + [dict(metric="legit test orders from unseen customers",
                ramped=round(ns_ramp, 4), flat_signup=round(ns_flat, 4)),
           dict(metric="median tenure, legit test orders (days)",
                ramped=round(ten_ramp, 1), flat_signup=round(ten_flat, 1)),
           dict(metric="COD orders, month 1",
                ramped=int(prof_ramp.cod_orders.iloc[0]),
                flat_signup=int(prof_flat.cod_orders.iloc[0])),
           dict(metric="COD orders, month 18",
                ramped=int(prof_ramp.cod_orders.iloc[-1]),
                flat_signup=int(prof_flat.cod_orders.iloc[-1]))])

    print("\n[flat] === temporal split: ramped vs flat signup ===")
    print(comp.to_string(index=False))
    print("\n[flat] === recall by ring type, flat-signup temporal ===")
    print(rt_flat.to_string(index=False))

    comp.to_csv(os.path.join(ARTIFACT_DIR, "flat_signup_diagnostic.csv"), index=False)
    rt_flat.to_csv(os.path.join(ARTIFACT_DIR, "flat_signup_recall_by_ring_type.csv"),
                   index=False)
    with open(os.path.join(ARTIFACT_DIR, "flat_signup_diagnostic.json"), "w") as f:
        json.dump({"status": "GENERATOR DIAGNOSTIC - not a headline result",
                   "ring_grouped_frozen_test_set_scored": False,
                   "note": "data/ is read for descriptive stats on the ramped "
                           "side only; nothing is fitted or thresholded on it",
                   "ramped_temporal": m_ramp, "flat_temporal": m_flat,
                   "flat_policy": pol_flat,
                   "unseen_customer_share_ramped": ns_ramp,
                   "unseen_customer_share_flat": ns_flat,
                   "median_tenure_ramped": ten_ramp,
                   "median_tenure_flat": ten_flat},
                  f, indent=2)
    print("\n[flat] wrote artifacts/flat_signup_diagnostic.{csv,json}")


if __name__ == "__main__":
    main()
