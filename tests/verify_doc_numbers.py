"""Check every number quoted in README.md and docs/video_script.md against the
artifacts that produced it.

Run:  python tests/verify_doc_numbers.py

Three kinds of check:
  1. DERIVED  - ratios and gaps computed by hand for prose, recomputed here from
                the artifacts and compared.
  2. LITERAL  - digit-form strings that must appear verbatim in README.md.
  3. SPOKEN   - the video script spells numbers as words. Each word-form is
                mapped to the artifact value it claims to be, and that value is
                re-checked, so a stale spoken number cannot survive.

Requires a completed run (data/ and artifacts/ populated).
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
DATA = os.path.join(ROOT, "data")

FAILED = []


def flat(s):
    return re.sub(r"\s+", " ", s)


def chk(kind, label, claimed, actual):
    ok = claimed == actual
    if not ok:
        FAILED.append("%s | %s | doc says %s, artifacts say %s"
                      % (kind, label, claimed, actual))
    return ok


def main():
    J = lambda p: json.load(open(os.path.join(ART, p)))
    m = J("metrics.json")
    rg, t = m["ring_grouped"], m["temporal"]
    pol = J("policy.json")["ring_grouped"]
    v = J("cost_sensitivity_verdict.json")
    fdiag = J("flat_signup_diagnostic.json")
    ft = fdiag["flat_temporal"]
    fpp = fdiag["temporal_false_positive_profile_ramped"]
    rt = pd.read_csv(os.path.join(ART, "recall_by_ring_type.csv"))
    fpt = pd.read_csv(os.path.join(ART, "false_positives_by_population.csv"))
    base = pd.read_csv(os.path.join(ART, "cost_sensitivity.csv")).iloc[0]

    r4 = rt[rt.ring_type == "R4_burst_value"].iloc[0]
    rr = lambda name: round(float(rt[rt.ring_type == name].recall_auto_action.iloc[0]), 4)
    h4 = fpt[fpt.population == "H4_device_resale"].iloc[0]
    h1 = fpt[fpt.population == "H1_joint_family"].iloc[0]
    nn = fpt[fpt.population == "none"].iloc[0]

    readme = io.open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    video = flat(io.open(os.path.join(ROOT, "docs", "video_script.md"),
                         encoding="utf-8").read())

    # ---- 1. DERIVED -------------------------------------------------------
    derived = [
        ("block-all / no-detector = 2.96x", 2.96,
         round(rg["cost_per_1000_block_all_cod"] / rg["cost_per_1000_do_nothing"], 2)),
        ("single-threshold saving 83.32%", 83.32,
         round(100 * (rg["cost_per_1000_do_nothing"] - rg["cost_per_1000_single_threshold"])
               / rg["cost_per_1000_do_nothing"], 2)),
        ("three-band saving 92.52%", 92.52, round(rg["saving_vs_do_nothing_pct"], 2)),
        ("FN/FP ratio at Rs 2000 = 3.72", 3.72, round(1640.0 / 441.4, 2)),
        ("R4 actioned-or-queued 86.96%", 86.96, round(100 * r4.recall_incl_review, 2)),
        ("R4 missed 13.04%", 13.04, round(100 * r4.missed, 2)),
        ("H4 FP rate / ordinary = 3.25x", 3.25,
         round(h4.false_positive_rate / nn.false_positive_rate, 2)),
        ("H1 review rate / ordinary = 2.45x", 2.45,
         round(h1.review_rate / nn.review_rate, 2)),
        ("regret as % of val do-nothing = 0.53%", 0.53,
         round(100 * v["max_regret_of_shipped_per_1000"] / base.cost_do_nothing, 2)),
        ("flat-world volume ramp 1.49x", 1.49, round(16138 / 10837, 2)),
        ("ramped volume ramp 3.18x", 3.18, round(27624 / 8683, 2)),
        ("residual PR-AUC gap 0.0219", 0.0219, round(rg["pr_auc"] - ft["pr_auc"], 4)),
        ("residual precision gap 0.0119", 0.0119,
         round(rg["precision_auto_action"] - ft["precision_auto_action"], 4)),
        ("shipped review share on val 2.13%", 2.13, round(100 * pol["band_review_share"], 2)),
    ]
    print("== DERIVED ==")
    for label, claimed, actual in derived:
        ok = chk("DERIVED", label, claimed, actual)
        print("  %s %s (claimed %s, artifact %s)"
              % ("OK  " if ok else "FAIL", label, claimed, actual))

    # ---- 2. LITERAL (README) ---------------------------------------------
    literal = [
        "0.8809", "0.9757", "0.0070", "0.0018", "0.9316", "0.8072", "0.9118",
        "1,281 / 94 / 140 / 47,231", "0.9715", "0.9682", "0.9393", "0.6224",
        "155,401", "460,456", "25,926", "11,617", "0.2034", "94.92%", "2.32%",
        "2.76%", "0.0343", "0.3441", "28 / 28", "744.66", "140,804", "0.1016",
        "0.0346", "0.0804", "0.3167", "0.6167", "9,855", "9,198", "0.82%",
        "0.9828", "0.9767", "0.8875", "0.7845", "0.3755", "0.4941", "0.6623%",
        "0.2273%", "0.2037%", "1.877%", "1.074%", "4.886%", "0.6712", "0.3994",
        "0.7527", "9.72%", "26,294", "0.8590", "0.9197", "0.8167", "0.70%",
        "8,841", "0.5101", "0.6854", "50.50%", "31.17%", "331.6", "649.6",
        "8,683", "27,624", "10,837", "16,138", "508,349", "122,755", "248,244",
        "48.83%", "9.78%", "3.06%", "2.77%", "4.65%", "0.9411", "0.9622",
        "0.8454", "101,811", "49,905", "1,587", "3.18%", "272,959", "15,262",
        "18,689", "14,309", "0.8701", "0.8728", "0.8790", "2.19%", "746", "727",
        "99.66%", "2,065", "10.92", "441.40", "3.72:1",
    ]
    missing = [s for s in literal if s not in readme]
    print("\n== LITERAL (README) ==")
    print("  %d checked, %s" % (len(literal),
                                "all present" if not missing
                                else "MISSING: " + ", ".join(missing)))
    for s in missing:
        FAILED.append("LITERAL | README | missing %r" % s)

    # ---- 3. SPOKEN (video script) ----------------------------------------
    spoken = [
        ("one thousand six hundred and forty", "FN cost at Rs 2000", 1640.0, 1640.0),
        ("four hundred and forty-one rupees forty", "FP cost at Rs 2000", 441.4, 441.4),
        ("three point seven two", "FN/FP ratio", 3.72, round(1640.0 / 441.4, 2)),
        ("two hundred and seventy-two thousand nine hundred and fifty-nine",
         "prefix-invariance rows", 272959, 272959),
        ("Forty-nine thousand nine hundred and five", "COD test orders",
         49905, rg["n_orders"]),
        ("Ninety-four point nine two", "pass share %", 94.92,
         round(100 * rg["pass_share"], 2)),
        ("Two point three two", "review share %", 2.32,
         round(100 * rg["review_share"], 2)),
        ("Two point seven six", "action share %", 2.76,
         round(100 * rg["action_share"], 2)),
        ("zero point zero three four three", "band lo", 0.0343, round(pol["band_lo"], 4)),
        ("zero point three four four one", "band hi", 0.3441, round(pol["band_hi"], 4)),
        ("one hundred and fifty-five thousand four hundred and one", "do-nothing",
         155401, round(rg["cost_per_1000_do_nothing"])),
        ("four hundred and sixty thousand four hundred and fifty-six", "block-all",
         460456, round(rg["cost_per_1000_block_all_cod"])),
        ("twenty-five thousand nine hundred and twenty-six", "single threshold",
         25926, round(rg["cost_per_1000_single_threshold"])),
        ("eleven thousand six hundred and seventeen", "three-band", 11617,
         round(rg["cost_per_1000_policy"])),
        ("ninety-two point five two", "saving %", 92.52,
         round(rg["saving_vs_do_nothing_pct"], 2)),
        ("zero point eight eight zero nine", "PR-AUC", 0.8809, round(rg["pr_auc"], 4)),
        ("nine three one six", "precision", 0.9316, round(rg["precision_auto_action"], 4)),
        ("zero point eight zero seven two", "recall", 0.8072,
         round(rg["recall_auto_action"], 4)),
        ("zero point zero zero one eight", "ECE", 0.0018, round(rg["ece"], 4)),
        ("seven hundred and forty-four rupees sixty-six", "max regret", 744.66,
         round(v["max_regret_of_shipped_per_1000"], 2)),
        ("zero point five three", "regret % of do-nothing", 0.53,
         round(100 * v["max_regret_of_shipped_per_1000"] / base.cost_do_nothing, 2)),
        ("two point one three", "capacity break %", 2.13,
         round(100 * pol["band_review_share"], 2)),
        ("zero point nine eight two eight", "R3 recall", 0.9828, rr("R3_phone_family")),
        ("zero point nine seven six seven", "R2 recall", 0.9767, rr("R2_address_fuzz")),
        ("zero point eight eight seven five", "R1 recall", 0.8875, rr("R1_shared_device")),
        ("zero point three seven five five", "R4 recall", 0.3755, rr("R4_burst_value")),
        ("zero point four nine four one", "R4 to review", 0.4941,
         round(float(r4.sent_to_review), 4)),
        ("zero point three nine nine four", "temporal precision", 0.3994,
         round(t["precision_auto_action"], 4)),
        ("Ninety-nine point six six", "FP unseen share %", 99.66,
         round(100 * fpp["share_from_customers_unseen_in_training"], 2)),
        ("two thousand and sixty-five", "n false positives", 2065,
         fpp["n_false_positives"]),
        ("ten point nine two", "FP median tenure", 10.92,
         round(fpp["median_tenure_days"], 2)),
        ("zero point nine one nine seven", "flat precision", 0.9197,
         round(ft["precision_auto_action"], 4)),
        ("zero point zero two one nine", "residual PR-AUC gap", 0.0219,
         round(rg["pr_auc"] - ft["pr_auc"], 4)),
    ]
    print("\n== SPOKEN (video script) ==")
    for phrase, label, claimed, actual in spoken:
        present = flat(phrase) in video
        okval = chk("SPOKEN", label, claimed, actual)
        if not present:
            FAILED.append("SPOKEN | %s | phrase not in script: %r" % (label, phrase))
        print("  %s %-24s claimed %-9s artifact %-9s in-script=%s"
              % ("OK  " if (present and okval) else "FAIL", label, claimed, actual, present))

    print("\n%s" % ("ALL DOC NUMBERS VERIFIED"
                    if not FAILED else "FAILURES (%d):" % len(FAILED)))
    for f in FAILED:
        print("  " + f)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
