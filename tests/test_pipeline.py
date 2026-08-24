"""Small, fast checks on the properties this project actually claims.

Run:  python tests/test_pipeline.py
Requires the pipeline to have been run at least once (needs data/ and artifacts/).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from costs import fn_cost, fp_cost  # noqa: E402
from features import (ALLOWED_ORDER_COLS, build_address_clusters,  # noqa: E402
                      build_phone_neighbours, normalise_address)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILED.append(name)


def test_address_normalisation():
    print("\naddress normalisation")
    same = ["Flat 4B, Sai Residency, MG Road",
            "Flt 4-B, SAI RES.  MG Road",
            "Apt 4 B , Sai Residency - MG Road",
            "No. 4B, Sai Residency, MG Road"]
    keys = {normalise_address(s) for s in same}
    check("four spellings of one flat collapse to one key", len(keys) == 1, str(keys))

    diff = normalise_address("Flat 5B, Sai Residency, MG Road")
    check("a different flat does NOT collapse", diff not in keys)

    other = normalise_address("Flat 4B, Green Towers, MG Road")
    check("a different building does NOT collapse", other not in keys)


def test_address_clustering_is_order_independent():
    print("\naddress clustering determinism (this is what makes it causal)")
    df = pd.DataFrame({
        "address_id": [1, 2, 3, 4],
        "raw_line": ["Flat 4B, Sai Residency, MG Road", "Flt 4-B, SAI RES.  MG Road",
                     "Flat 9C, Green Towers, Link Road", "Apt 4 B, Sai Residency, MG Road"],
        "pincode": [110001, 110001, 110002, 110001]})
    c1, _ = build_address_clusters(df)
    c2, _ = build_address_clusters(df.iloc[::-1].reset_index(drop=True))
    grp1 = {a: sorted(k for k, v in c1.items() if v == c1[a]) for a in c1}
    grp2 = {a: sorted(k for k, v in c2.items() if v == c2[a]) for a in c2}
    check("cluster MEMBERSHIP is independent of row order", grp1 == grp2)
    check("the three variants of 4B group together", grp1[1] == [1, 2, 4], str(grp1[1]))


def test_phone_neighbours():
    print("\nphone adjacency")
    ph = pd.DataFrame({"phone_id": [1, 2, 3, 4],
                       "msisdn": ["7000000000", "7000000001", "7000000002", "7000009999"]})
    nb = build_phone_neighbours(ph, window=5)
    check("consecutive numbers are neighbours", set(nb.get(1, [])) == {2, 3}, str(nb.get(1)))
    check("a distant number has no neighbours", 4 not in nb)


def test_cost_asymmetry():
    print("\ncost model")
    v = 2000.0
    fn, fp = float(fn_cost(v)), float(fp_cost(v))
    check("a miss costs more than a false alarm", fn > fp, "fn=%.0f fp=%.0f" % (fn, fp))
    check("fn cost on a 2000 order is 1640", abs(fn - 1640.0) < 1.0, "%.2f" % fn)
    check("fp cost on a 2000 order is ~441", abs(fp - 441.4) < 1.0, "%.2f" % fp)
    check("fn cost scales with order value",
          float(fn_cost(20000.0)) > 5 * float(fn_cost(2000.0)) * 0.5)


def test_no_ground_truth_in_feature_inputs():
    print("\nallow-list blocks ground truth")
    banned = {"is_abuse", "is_abuse_true", "ring_id", "ring_type", "household_id",
              "hn_type", "return_flag", "delivered_flag", "building_id",
              "phone_family_id", "segment"}
    leaked = banned & ALLOWED_ORDER_COLS
    check("no ground-truth / post-outcome column is allow-listed", not leaked, str(leaked))


def test_artifacts_consistent():
    print("\nartifact consistency (requires a completed run)")
    mp = os.path.join(ROOT, "artifacts", "metrics.json")
    pp = os.path.join(ROOT, "artifacts", "policy.json")
    if not (os.path.exists(mp) and os.path.exists(pp)):
        print("  SKIP  run `python run_all.py` first")
        return
    m = json.load(open(mp))
    pol = json.load(open(pp))
    rg = m["ring_grouped"]
    check("test-set bands equal the validation-selected bands",
          abs(rg["band_hi"] - pol["ring_grouped"]["band_hi"]) < 1e-12
          and abs(rg["band_lo"] - pol["ring_grouped"]["band_lo"]) < 1e-12)
    check("precision recomputes from tp/fp",
          abs(rg["precision_auto_action"] - rg["tp"] / (rg["tp"] + rg["fp"])) < 1e-9)
    check("recall recomputes from tp and positives",
          abs(rg["recall_auto_action"] - rg["tp"] / rg["n_abuse"]) < 1e-9)
    check("band shares sum to 1",
          abs(rg["pass_share"] + rg["review_share"] + rg["action_share"] - 1.0) < 1e-9)
    check("the policy beats doing nothing",
          rg["cost_per_1000_policy"] < rg["cost_per_1000_do_nothing"])
    check("review share respects the capacity constraint on the primary split",
          rg["review_share"] <= pol["ring_grouped"]["review_capacity_constraint"])
    rt = pd.read_csv(os.path.join(ROOT, "artifacts", "recall_by_ring_type.csv"))
    check("per-ring-type recall is reported for all five signatures",
          sum(1 for x in rt.ring_type if x.startswith("R")) == 5, str(list(rt.ring_type)))


def main():
    test_address_normalisation()
    test_address_clustering_is_order_independent()
    test_phone_neighbours()
    test_cost_asymmetry()
    test_no_ground_truth_in_feature_inputs()
    test_artifacts_consistent()
    print("\n%s" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
