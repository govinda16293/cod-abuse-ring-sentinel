"""Assemble the modelling frame.

One place decides what the SCOREABLE POPULATION is, so the number can never
drift between training, thresholding and reporting.

Scoreable population = COD orders. The detector's only actions are "force
prepay" and "hold for review"; neither is meaningful on an order that is already
prepaid. Quoting the abuse rate over all orders would halve it for free, so
every rate in this repo is quoted over COD orders and labelled as such.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLUMNS, STABLE_FEATURE_COLUMNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")

LABEL = "is_abuse"              # observed label - what a merchant would actually have
LABEL_TRUE = "is_abuse_true"    # noiseless label - ceiling check only, never trained on


def active_features():
    """Which feature set the pipeline is using.

    Chosen by src/experiment_featureset.py on an out-of-period fold that ends at
    month 12. Resolved here so training, thresholding, evaluation and the UI can
    never disagree about it.
    """
    import json
    path = os.path.join(ARTIFACT_DIR, "featureset_choice.json")
    if os.path.exists(path):
        with open(path) as f:
            sel = json.load(f).get("selected")
        if sel == "C_stable_only":
            return STABLE_FEATURE_COLUMNS
    return FEATURE_COLUMNS


def load_modelling_frame(data_dir=DATA_DIR, cod_only=True):
    feats = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
    truth = pd.read_parquet(os.path.join(data_dir, "ground_truth.parquet"))
    orders = pd.read_parquet(os.path.join(data_dir, "orders.parquet"),
                             columns=["order_id", "payment_mode", "order_ts", "customer_id",
                                      "value_inr", "category"])
    df = feats.merge(truth, on="order_id", how="inner").merge(
        orders, on="order_id", how="inner", suffixes=("", "_o"))
    if cod_only:
        df = df.loc[df.payment_mode == "COD"].reset_index(drop=True)
    return df


def xy(df, label=LABEL, cols=None):
    X = df[cols if cols is not None else active_features()].to_numpy(dtype=np.float32)
    y = df[label].to_numpy(dtype=np.int8)
    return X, y


def frozen_test_ids(data_dir=DATA_DIR):
    return set(np.load(os.path.join(data_dir, "test_order_ids.npy")).tolist())
