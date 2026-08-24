"""Phase 3 - calibrated gradient-boosted scorer.

Model choice, in one line each:

* LightGBM, not a GNN. The graph structure that matters here is already
  summarised into ~15 component-level features computed as-of; a boosted tree
  over those is defensible feature-by-feature under questioning, trains in
  seconds, and gives exact per-prediction attribution via TreeSHAP. A GNN would
  add message passing over an identity graph that is mostly stars and small
  cliques, and would make the causality guarantee much harder to prove, because
  neighbourhood aggregation has to be time-masked at every hop.
* Isotonic calibration, not Platt. The validation fold has ~100k rows and ~3k
  positives, which is enough for a non-parametric monotone fit, and the raw
  boosted-tree scores are not sigmoid-shaped near 0, where all our thresholds
  live.
* No class reweighting. Reweighting distorts the output scale, and the whole
  point of Phase 4 is to threshold a probability that means what it says.
  Imbalance is handled by calibrating afterwards and by ranking metrics (PR-AUC).
* Calibrator is fit on the VALIDATION fold only, with cv="prefit". Fitting it on
  training data would calibrate against scores the model has already memorised.

Two models are trained: `primary` on the ring-grouped split, `temporal` on
months 1-12. They are separate artefacts and never mixed.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL  # noqa: E402
from dataset import (ARTIFACT_DIR, LABEL, active_features,  # noqa: E402
                     load_modelling_frame, xy)
from features import FEATURE_COLUMNS  # noqa: E402


def _fit(Xtr, ytr, Xva, yva):
    clf = LGBMClassifier(
        n_estimators=MODEL.n_estimators, learning_rate=MODEL.learning_rate,
        num_leaves=MODEL.num_leaves, min_child_samples=MODEL.min_child_samples,
        subsample=MODEL.subsample, subsample_freq=MODEL.subsample_freq,
        colsample_bytree=MODEL.colsample_bytree, reg_lambda=MODEL.reg_lambda,
        random_state=MODEL.seed, n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(clf, method=MODEL.calibration_method, cv="prefit")
    cal.fit(Xva, yva)
    return clf, cal


def train_split(df, split_col, train_val="train", val_val="val", test_val="test",
                val_from_train_frac=None, seed=MODEL.seed):
    """Fit on the train fold, calibrate on val. Test is never touched here."""
    tr = df.loc[df[split_col] == train_val]
    va = df.loc[df[split_col] == val_val]
    if len(va) == 0:
        # temporal split has no natural val fold; carve one out of train by
        # GROUP so ring integrity survives, using the last 2 months of train.
        cut = tr.order_ts.quantile(1.0 - (val_from_train_frac or 0.2))
        va_groups = set(tr.loc[tr.order_ts >= cut, "group_key"].unique().tolist())
        va = tr.loc[tr.group_key.isin(va_groups) & (tr.order_ts >= cut)]
        tr = tr.loc[~tr.group_key.isin(va_groups)]
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    clf, cal = _fit(Xtr, ytr, Xva, yva)
    p_va = cal.predict_proba(Xva)[:, 1]
    info = dict(n_train=int(len(tr)), n_val=int(len(va)),
                train_positives=int(ytr.sum()), val_positives=int(yva.sum()),
                val_pr_auc=float(average_precision_score(yva, p_va)),
                val_roc_auc=float(roc_auc_score(yva, p_va)),
                val_base_rate=float(yva.mean()))
    return clf, cal, tr, va, p_va, info


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_modelling_frame()
    COLS = active_features()
    print("[train] feature set: %d columns" % len(COLS))
    print("[train] scoreable population (COD orders): %d, observed abuse rate %.4f"
          % (len(df), df[LABEL].mean()))

    out = {}

    # ---- primary: ring-grouped split ---------------------------------------
    clf, cal, tr, va, p_va, info = train_split(df, "split")
    out["primary"] = info
    print("[train] primary  ", json.dumps(info))
    with open(os.path.join(ARTIFACT_DIR, "model_primary.pkl"), "wb") as f:
        pickle.dump({"booster": clf, "calibrator": cal, "features": COLS}, f)
    va_out = va[["order_id", "group_key", "ring_type", LABEL, "value_inr"]].copy()
    va_out["p"] = p_va
    va_out.to_parquet(os.path.join(ARTIFACT_DIR, "val_predictions_primary.parquet"),
                      index=False)

    # ---- secondary: temporal split -----------------------------------------
    dt = df.loc[df.temporal_split != "drop"]
    clf_t, cal_t, tr_t, va_t, p_va_t, info_t = train_split(dt, "temporal_split")
    out["temporal"] = info_t
    print("[train] temporal ", json.dumps(info_t))
    with open(os.path.join(ARTIFACT_DIR, "model_temporal.pkl"), "wb") as f:
        pickle.dump({"booster": clf_t, "calibrator": cal_t, "features": COLS}, f)
    vt = va_t[["order_id", "group_key", "ring_type", LABEL, "value_inr"]].copy()
    vt["p"] = p_va_t
    vt.to_parquet(os.path.join(ARTIFACT_DIR, "val_predictions_temporal.parquet"), index=False)

    imp = pd.DataFrame({"feature": COLS,
                        "gain": clf.booster_.feature_importance("gain"),
                        "split": clf.booster_.feature_importance("split")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False)
    print("[train] top features by gain:")
    print(imp.head(15).to_string(index=False))

    with open(os.path.join(ARTIFACT_DIR, "train_info.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
