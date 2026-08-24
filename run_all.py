"""One script, clean machine, end to end.

    python run_all.py            # full pipeline, then print where things are
    python run_all.py --ui       # ...and launch the Streamlit UI
    python run_all.py --quick    # 10x smaller dataset, for a smoke test

Stage order is not arbitrary. The leakage check runs BEFORE any model is
trained, so a leaky feature can never reach a metric. Thresholds are selected
before evaluation and are read back from disk rather than passed in memory, so
the test set cannot influence them even by accident.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    ("Phase 1  generate synthetic data + freeze/hash test set", "src/generate_data.py"),
    ("Phase 2  build point-in-time features", "src/features.py"),
    ("Phase 2b leakage audit + causality proof", "src/leakage_audit.py"),
    ("Phase 3a select feature set on an out-of-period fold (never test)",
     "src/experiment_featureset.py"),
    ("Phase 3  train calibrated scorers (ring-grouped + temporal)", "src/train.py"),
    ("Phase 4  cost-based threshold and band selection", "src/threshold.py"),
    ("Phase 5  evaluate ONCE on the frozen test set", "src/evaluate.py"),
]


def run(script, env):
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(ROOT, script)], cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit("\nFAILED: %s (exit %d)" % (script, r.returncode))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", action="store_true", help="launch Streamlit when done")
    ap.add_argument("--quick", action="store_true", help="10x smaller dataset")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.quick:
        env["SENTINEL_QUICK"] = "1"
        print("[run] QUICK MODE: 12,000 customers / 50,000 orders. Metrics from this "
              "run are NOT the reported metrics.\n")

    total = 0.0
    for i, (label, script) in enumerate(STAGES, 1):
        print("\n" + "=" * 78)
        print("[%d/%d] %s" % (i, len(STAGES), label))
        print("=" * 78, flush=True)
        total += run(script, env)

    print("\n" + "=" * 78)
    print("done in %.1fs" % total)
    print("""
  data/dataset_stats.json              population and base rates
  data/test_set.sha256                 frozen test-set hash
  docs/leakage_audit.md                per-feature placement-time audit
  artifacts/metrics.json               held-out results, both splits
  artifacts/recall_by_ring_type.csv    recall by ring signature R1-R5
  artifacts/false_positives_by_population.csv
  artifacts/policy.json                thresholds, selected on validation
  artifacts/figures/                   PR, reliability, cost curves

  streamlit run app/streamlit_app.py   review UI
  python tests/test_pipeline.py        property checks
""")
    if args.ui:
        subprocess.run([PY, "-m", "streamlit", "run",
                        os.path.join(ROOT, "app", "streamlit_app.py")], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
