"""Improvement pass - choosing a feature set WITHOUT touching the test set.

The problem: the drift that wrecked the temporal number lives in months 13-18,
which are test. The temporal model's own validation fold is carved out of
months 1-12, so it is in-period and shows no drift at all. Selecting on it would
be selecting on the wrong thing.

The protocol here uses only data up to month 12:

    fit        months 1-8
    calibrate  month  9
    select on  months 10-12      <- an out-of-period fold with a real time gap

Rings are kept non-crossing at every boundary. Whichever feature set wins here
is then trained on the full months 1-12 and evaluated once on months 13-18.
Nothing in this file reads the frozen test set.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import COST, MODEL  # noqa: E402
from costs import policy_cost  # noqa: E402
from dataset import ARTIFACT_DIR, LABEL, load_modelling_frame  # noqa: E402
from features import FEATURE_COLUMNS, STABLE_FEATURE_COLUMNS  # noqa: E402
from threshold import sweep_bands  # noqa: E402

START = pd.Timestamp("2025-01-01")


def month_slice(df, m0, m1):
    lo = START + pd.DateOffset(months=m0)
    hi = START + pd.DateOffset(months=m1)
    return df.loc[(df.order_ts >= lo) & (df.order_ts < hi)]


def drop_crossing_rings(fold, earlier):
    """A ring seen in an earlier fold cannot appear in a later one."""
    seen = set(earlier.loc[earlier.ring_id >= 0, "ring_id"].unique().tolist())
    return fold.loc[~((fold.ring_id >= 0) & (fold.ring_id.isin(seen)))]


def run_variant(df, cols, name):
    fit = month_slice(df, 0, 8)
    cal = drop_crossing_rings(month_slice(df, 8, 9), fit)
    sel = drop_crossing_rings(month_slice(df, 9, 12), pd.concat([fit, cal]))

    clf = LGBMClassifier(
        n_estimators=MODEL.n_estimators, learning_rate=MODEL.learning_rate,
        num_leaves=MODEL.num_leaves, min_child_samples=MODEL.min_child_samples,
        subsample=MODEL.subsample, subsample_freq=MODEL.subsample_freq,
        colsample_bytree=MODEL.colsample_bytree, reg_lambda=MODEL.reg_lambda,
        random_state=MODEL.seed, n_jobs=-1, verbose=-1)
    clf.fit(fit[cols].to_numpy(np.float32), fit[LABEL].to_numpy())
    calib = CalibratedClassifierCV(clf, method=MODEL.calibration_method, cv="prefit")
    calib.fit(cal[cols].to_numpy(np.float32), cal[LABEL].to_numpy())

    y = sel[LABEL].to_numpy()
    p = calib.predict_proba(sel[cols].to_numpy(np.float32))[:, 1]
    v = sel.value_inr.to_numpy()

    # pick bands on the CALIBRATION month, apply to the selection months, so the
    # comparison includes the threshold-transfer behaviour we actually care about
    yc = cal[LABEL].to_numpy()
    pc = calib.predict_proba(cal[cols].to_numpy(np.float32))[:, 1]
    bands = sweep_bands(yc, pc, cal.value_inr.to_numpy())
    feasible = bands.loc[bands.review_share <= COST.max_review_share]
    bb = feasible.loc[feasible.cost_per_1000.idxmin()]
    pol = policy_cost(y, p, v, float(bb.lo), float(bb.hi))

    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec >= 0.70
    p_at_r70 = float(prec[np.flatnonzero(ok)[-1]]) if ok.any() else float("nan")

    tp, fp = pol["tp"], pol["fp"]
    return dict(
        variant=name, n_features=len(cols),
        fit_n=len(fit), cal_n=len(cal), sel_n=len(sel),
        sel_base_rate=float(y.mean()),
        pr_auc=float(average_precision_score(y, p)),
        p_at_r70=p_at_r70,
        precision=float(tp / (tp + fp)) if (tp + fp) else 0.0,
        recall=float(tp / max(int(y.sum()), 1)),
        review_share=pol["review_share"],
        cost_per_1000=pol["cost_per_1000"],
        band_lo=float(bb.lo), band_hi=float(bb.hi),
    )


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_modelling_frame()
    variants = [
        ("A_pre_improvement", [c for c in FEATURE_COLUMNS
                               if c not in ("cust_orders_per_day", "dev_orders_per_day",
                                            "addr_orders_per_day", "addr_age_days",
                                            "comp_orders_per_member_per_day",
                                            "pin_orders_share", "pin_cust_share",
                                            "comp_orders_share")]),
        ("B_plus_rate_features", FEATURE_COLUMNS),
        ("C_stable_only", STABLE_FEATURE_COLUMNS),
    ]
    rows = [run_variant(df, cols, name) for name, cols in variants]
    out = pd.DataFrame(rows)
    print("\nOut-of-period selection fold (months 10-12), fit on 1-8, calibrated on 9")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(ARTIFACT_DIR, "featureset_experiment.csv"), index=False)
    best = out.loc[out.cost_per_1000.idxmin(), "variant"]
    print("\nlowest expected cost on the out-of-period fold: %s" % best)
    with open(os.path.join(ARTIFACT_DIR, "featureset_choice.json"), "w") as f:
        json.dump({"selected": best, "criterion":
                   "minimum expected rupee cost on an out-of-period fold (months 10-12); "
                   "test set not consulted", "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
