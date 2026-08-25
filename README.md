# COD abuse-ring sentinel

A defence-only detector for **cash-on-delivery return abuse** in Indian
e-commerce. It ingests an order stream, builds a point-in-time customer /
device / address / phone graph, scores each COD order's abuse risk, picks its
decision threshold by **expected rupee cost** rather than by F1, and emits a
reviewable explanation for every flag together with an honest list of what it
could not resolve.

Razorpay AI Buildathon, Track 02 (AI Risk Manager). Everything below is measured
on a held-out test set that was generated, hashed and frozen before any model
existed.

---

## 1. The problem

A customer orders a ₹4,000 pair of shoes cash-on-delivery, receives them, files a
"wrong item" return, and gets a cash refund at the door. The merchant has paid
forward shipping, reverse pickup, and handling, and the item that comes back is
worthless or absent.

Done once, that is a bad customer. Done by **eleven accounts sharing two devices
and one flat, across six weeks**, it is a ring — and the loss is systematic. The
accounts individually look unremarkable; the pattern only exists between them.

## 2. Why it costs merchants money

Per abusive COD order, using the cost model in [`src/config.py`](src/config.py):

| Component | Value |
|---|---|
| Forward shipping | ₹75 |
| Reverse pickup | ₹85 |
| Ops handling | ₹40 |
| Product loss (72% of order value — not 100%, some goods return resaleable) | 0.72 × V |
| **False negative cost** | **₹200 + 0.72 V** |

And the other direction, because blocking a real customer is not free:

| Component | Value |
|---|---|
| Margin lost if they abandon on forced prepay (34% × 18% margin) | 0.061 × V |
| Churn (22% never return × ₹1,450 residual value) | ₹319 |
| **False positive cost** | **₹319 + 0.061 V** |

Every constant above is re-derived against published Indian 3PL rate cards and
category margin benchmarks in **[docs/cost_model.md](docs/cost_model.md)**, with
sources. The honest split: of eleven constants, four are sourced to published
figures, two are derived from published labour rates, and **five have no public
figure at all** and are stated assumptions. Two are flagged as wrong-ish and left
unchanged — reverse shipping at ₹85 is 113% of forward, above the cited 80–100%
RTO band, and `gross_margin_rate` is really a *contribution* margin, not a gross
one. COD collection fees ("₹40 or 2%, whichever is higher") are omitted entirely,
which understates the cost of a miss.

That is why the defence of this result is not "the constants are right" — it is
**[docs/cost_sensitivity.md](docs/cost_sensitivity.md)**, summarised in §5.

On a ₹2,000 order that is **₹1,640 for a miss against ₹441 for a false alarm** —
a 3.7:1 asymmetry. F1 implicitly assumes 1:1. That single fact is why this repo
selects its threshold on a cost curve, and it moves the operating point a long
way from where an F1-optimal threshold would sit.

Measured on the frozen test set: doing nothing costs **₹155,401 per 1,000 COD
orders**. Switching COD off entirely costs **₹460,456**. This detector costs
**₹11,617** — a **92.5% reduction** against no detector.

## 3. What I built

```
orders ─┬─► point-in-time feature replay ──► LightGBM ──► isotonic ──► P(abuse)
        │   (as-of graph + behaviour)                     calibration      │
returns ┘                                                                  ▼
                                        cost curve in ₹ ──► 3-band decision policy
                                                                           │
                                            TreeSHAP attribution ──────────┤
                                                                           ▼
                                            LLM writes the justification (never the score)
```

Six stages, one command. See [ARCHITECTURE.md](ARCHITECTURE.md) for the detail.

**The one design decision everything else hangs off:** every feature for an order
placed at time *T* is computed from data strictly before *T*. Orders and returns
are replayed as timestamped events against a mutable state object; a return
becomes visible only at its `return_ts`, not at its `order_ts`. There is no
global graph pass followed by a split, because a global graph contains edges
created by test-set orders.

This is **verified, not asserted**. `assert_prefix_invariance` deletes every row
after a cutoff date, recomputes all features from the truncated world, and
requires bit-identical output. It runs *before* any model is trained.

```
[leak] PASS - all 63 features bit-identical on 272,959 pre-cutoff orders
```

## 4. The data

Fully synthetic, seeded, regenerable in ~17s. No scraped, purchased or real
customer data; phone numbers and addresses are drawn from token pools.

| | |
|---|---|
| Orders | 508,349 over 18 months (Jan 2025 – Jun 2026) |
| Customers | 122,755 |
| **Scoreable population (COD orders)** | **248,244 (48.8% of orders)** |
| Abuse rings | 512 |
| Hard-negative customers | 9.8% of the base |

### Base rates — stated over the population the detector actually scores

The detector's only actions are "force prepay" and "hold for review". Neither is
meaningful on an order that is already prepaid, so **prepaid orders are never
scored** and every rate below is quoted over COD orders.

| Rate | Value |
|---|---|
| **Abuse rate, COD orders (observed labels)** | **3.06%** |
| Abuse rate, COD orders (noiseless labels) | 3.00% |
| Abuse rate, all orders — *the flattering number, not used* | 1.58% |
| **Abuse rate, COD-active customers** | **2.77%** |
| Abuse orders on prepaid (out of scope, unaddressed loss) | 405 |

### Label noise

Set to a non-zero value deliberately; perfectly clean labels are a tell.

- 3.0% of true-abuse COD orders are recorded as legitimate (217 orders) — ops misses.
- Injected false positives equal 5.0% of the positive count (373 orders), 70% of
  them drawn from hard-negative households, because those are the cases a real
  ops team actually gets wrong.
- Net: **4.65% of positive labels are wrong.** Training and all headline metrics
  use the *observed* (noisy) labels, because that is what a merchant would have.

**Ceiling check.** The same model evaluated against noiseless labels scores
PR-AUC 0.941 / precision 0.962 / recall 0.845, versus 0.881 / 0.932 / 0.807
observed. So roughly 6 points of PR-AUC is label noise, not model error.

### Ring signatures and hard negatives

| Abuse ring | Shares | | Hard negative | Trips the naive rule |
|---|---|---|---|---|
| R1 shared_device | 1–2 devices, distinct addresses | | H1 joint_family | "same address AND device" |
| R2 address_fuzz | one flat, many spellings | | H2 hostel_pincode | "pincode concentration" |
| R3 phone_family | adjacent MSISDNs | | H3 high_returner | "return rate > 40%" |
| R4 burst_value | **nothing** — behaviour only | | H4 device_resale | "device on 2 accounts" |
| R5 hybrid | partial mix + benign filler | | | |

Legitimate joint families also get fuzzed address variants, and half of all
ordinary customers' second addresses are a re-typing of their own. Without that,
"this door has been spelled five ways" would separate R2 from real families
perfectly — a signal that exists nowhere outside a generator.

## 5. Metrics

**Frozen test set**, `sha256 = 06a9b5c9…73ad98`, 101,811 orders of which 49,905
are COD and scoreable. Thresholds selected on validation and read back from
`artifacts/policy.json`; never re-tuned.

### Headline — ring-grouped split (primary)

| Metric | Value |
|---|---|
| COD orders scored | 49,905 |
| Abuse orders | 1,587 (3.18%) |
| **PR-AUC** | **0.8809** |
| ROC-AUC | 0.9757 |
| Brier | 0.0070 |
| Expected calibration error | 0.0018 |
| **Precision (auto-action band)** | **0.9316** |
| **Recall (auto-action band)** | **0.8072** |
| Recall including review band | 0.9118 |
| TP / FP / FN | 1,281 / 94 / 140 |

### Precision at fixed recall

| Recall | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|
| Precision | 0.972 | 0.972 | 0.968 | 0.939 | 0.622 |

Precision holds above 0.93 all the way to 80% recall, then falls off a cliff
between 80% and 90%. That cliff is R4_burst_value (see below).

### Cost table (₹ per 1,000 COD orders)

| Policy | Cost | vs. doing nothing |
|---|---|---|
| No detector | 155,401 | — |
| Block all COD | 460,456 | 3.0x worse |
| Single cost-optimal threshold (0.2034) | 25,926 | −83.3% |
| **Three-band policy (shipped)** | **11,617** | **−92.5%** |

### Where the volume lands

| Band | Range | Share of COD orders | Action |
|---|---|---|---|
| Pass | p < 0.0343 | 94.92% | ship as COD |
| Review | 0.0343 <= p < 0.3441 | 2.32% | human queue |
| Auto-action | p >= 0.3441 | 2.76% | force prepay |

**2.32% to manual review**, against a 5% capacity constraint imposed during band
selection. A system that sent 40% to humans would not be deployable, and the
constraint lives in the optimiser, not in the write-up.

### Recall by ring signature

The number the brief asks for, and the one a blended recall hides:

| Ring type | Abuse orders | Recall (auto-action) | Sent to review | Missed |
|---|---|---|---|---|
| R3_phone_family | 407 | **0.983** | 0.010 | 0.007 |
| R2_address_fuzz | 429 | **0.977** | 0.014 | 0.009 |
| R1_shared_device | 311 | **0.887** | 0.039 | 0.074 |
| R5_hybrid | 116 | **0.784** | 0.147 | 0.069 |
| R4_burst_value | 253 | **0.375** | 0.494 | 0.130 |
| *(label noise, no ring)* | 71 | 0.000 | 0.028 | 0.972 |

**This inverts the expectation in the brief, and the reason matters.** R2 address
fuzzing was predicted to be hard and R4 burst-value easy. The opposite happened.

- R2 is caught (0.977) because address normalisation re-clusters the spellings,
  giving a strong graph edge. Its ceiling is my synonym dictionary — see
  [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). This number was 0.918 until a
  unit test caught a bug where `apt` was expanded into a building-name token
  instead of stripped as a prefix, splitting clusters (see [LOG.md](LOG.md)).
- **R4 is the hardest (0.375) because it shares no identifier at all.** By
  construction it has no device, address or phone link. All that is left is "new
  account, COD, above-average value" — which is also an exact description of a
  legitimate first-time COD customer. The information needed to separate them is
  not in the data at checkout time. The model is not failing to learn; there is
  nothing there to learn. Note that 49.4% of R4 orders *are* routed to human
  review, so the system triages them rather than missing them silently.

The bottom row is a sanity check: the 71 injected label-noise positives are
correctly *not* chased.

### False positives by legitimate population

| Population | Legit COD orders | FP rate | To review | FPs |
|---|---|---|---|---|
| H4_device_resale | 604 | 0.662% | 3.15% | 4 |
| H1_joint_family | 1,760 | 0.227% | 4.89% | 4 |
| ordinary customers | 42,229 | 0.204% | 1.99% | 86 |
| H3_high_returner | 2,979 | **0.000%** | 1.07% | 0 |
| H2_hostel_pincode | 746 | **0.000%** | 1.88% | 0 |

The hard negatives are doing their job: resold devices are ~3x more likely to be
flagged than an ordinary customer, and joint families are routed to review at
2.5x the ordinary rate. Genuine high-return shoppers and hostel residents produce
**zero** false positives — the model is not using "high return rate" or
"pincode concentration" as a lazy proxy.

### Second evaluation — temporal holdout (months 13–18)

| Metric | Ring-grouped | Temporal |
|---|---|---|
| Base rate | 3.18% | 1.58% |
| PR-AUC | 0.8809 | **0.6712** |
| Precision (auto-action) | 0.9316 | **0.3994** |
| Recall (auto-action) | 0.8072 | 0.7527 |
| Review share | 2.32% | **9.72%** |
| Cost / 1,000 | ₹11,617 | ₹26,294 |

The base-rate gap is not a mystery and is worth stating up front: the
ring-grouped test samples groups uniformly across all 18 months so it inherits the
dataset-wide COD abuse rate (3.06%), whereas the temporal test covers only months
13–18, where benign COD volume has ramped 3.18x while ring campaigns stayed
uniform (dropping the rate to 2.19%) and then loses a further 746 orders — 727 of
them abusive — when straddling rings are removed to keep rings non-crossing,
landing at 1.58%.

**The two splits disagree badly.** Recall barely moves (0.807 -> 0.753).
Precision falls by 2.3x, and the review queue blows through its 5% capacity
constraint. I originally reported this as the headline finding of the project.
**A follow-up diagnostic showed most of it was my generator's fault** — see
"Flat-signup diagnostic" below, and read that before quoting these numbers.

I diagnosed it rather than reporting it as a mystery. On the temporal test set:

- **99.3%** of false-positive orders come from customers never seen in training,
  against 50.5% for legit test orders generally.
- Median tenure of a false positive: **12.6 days** (legit median: 331.6 days).
- Median `comp_size` of a false positive: **1.00** — *no identity links at all*.
- The 99th percentile of legit-order scores moves from 0.068 to 0.694.

So the failure is not overfitting and not a leak. The model learned in months
1–12 that "new account + COD + above-average value" is abuse — which is the R4
signature — and in months 13–18 it meets a far larger population of legitimate
brand-new accounts and flags them. It is the same root cause as the R4 recall
number, seen from the other side.

### Flat-signup diagnostic — how much of that was real?

In the headline generator every customer draws the same expected order count
regardless of signup date, so a customer joining five days before the window ends
crams all of them into those five days. That inflates both order volume and the
supply of accounts that are simultaneously brand-new and ordering fast — which is
the R4 signature the model latched onto.

So I regenerated **once** with `flat_signup=True`, which scales each customer's
order rate by the fraction of the window they were present for. Same seed, same
rings, same hard negatives, same label noise, same feature code, same model, same
band-selection procedure. It writes to `data_flat/`. It does read `data/` for the
ramped side's descriptive statistics, but **it never scores the ring-grouped
frozen test set, fits nothing on it, and selects no threshold from it** —
`data/test_set.sha256` is byte-identical before and after a run.

| Temporal split | Ramped (headline) | Flat signup | Ring-grouped, for reference |
|---|---|---|---|
| Base rate | 1.58% | 1.92% | 3.18% |
| PR-AUC | 0.6712 | **0.8590** | 0.8809 |
| Precision (auto-action) | 0.3994 | **0.9197** | 0.9316 |
| Recall (auto-action) | 0.7527 | 0.8167 | 0.8072 |
| Review share | 9.72% | **0.70%** | 2.32% |
| ₹ / 1,000 | 26,294 | 8,841 | 11,617 |
| R4_burst_value recall | 0.5101 | 0.6854 | 0.3755 |
| Legit test orders from unseen customers | 50.5% | 31.2% | — |
| Median tenure, legit test orders | 331.6 d | 649.6 d | — |
| COD orders, month 1 → month 18 | 8,683 → 27,624 | 10,837 → 16,138 | — |

**This is a generator diagnostic, not a headline result**, and it does not change
a single number in the primary table above. But it changes what those numbers
*mean*, so:

**Correction.** I previously wrote that the temporal collapse was real in
direction and merely inflated in magnitude. That was wrong, and the diagnostic
says so: with exposure-adjusted ordering, temporal precision recovers to 0.9197
against a ring-grouped 0.9316, and the review queue drops to 0.70% — comfortably
inside its constraint. **Most of the collapse was an artefact of how I built the
data, not a property of the detector.**

What survives is small and real: PR-AUC 0.8590 vs 0.8809 and precision 0.9197 vs
0.9316. So roughly 2 points of PR-AUC of genuine degradation over a six-month
horizon, not the 21 points originally reported. Residual volume still ramps 1.49x
in the flat world (10.8k to 16.1k), because customers who sign up mid-window still
contribute partial windows, so even that 2 points is an upper bound.

Reproduce with `python src/diagnostic_flat_signup.py`.

### Feature-set selection, done without touching test

I added 8 scale-free rate features to fix counter drift, then had to choose
between feature sets. That could not be done on the temporal model's own
validation fold, because that fold sits inside months 1–12 and contains none of
the drift. The protocol uses only data up to month 12: fit on months 1–8,
calibrate on month 9, select on months 10–12, rings non-crossing throughout.

| Variant | Features | Cost/1000 on the selection fold | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| A pre-improvement | 55 | ₹15,262 | 0.8701 | 0.9253 | 0.8320 |
| B + rate features | 63 | ₹18,689 | 0.8728 | 0.9177 | 0.8230 |
| **C stable only (shipped)** | **49** | **₹14,309** | **0.8790** | **0.9367** | **0.7885** |

An earlier version of this comparison was also run on the test set, before the
address-normalisation bug was found. Those numbers are superseded and are not
quoted here; the record is in [LOG.md](LOG.md).

**Disclosure:** the frozen test set has been scored three times across the
project — pre-improvement, post-improvement, and after the address bug fix. Only
the last is reported. No threshold, band, feature set or hyperparameter was ever
selected using it.

### Cost sensitivity — does the conclusion survive being wrong?

The whole headline rests on constants I asserted, five of which have no published
figure. So every one of them is swept, re-running threshold and band selection
from scratch on the **validation fold** each time (the frozen test set is not
opened by this analysis). Full table: [docs/cost_sensitivity.md](docs/cost_sensitivity.md).

Swept: the fixed and variable halves of both the false-negative and false-positive
cost at ±25% and ±50%, reviewer accuracy 70–95%, and review capacity 2–10%.
**28 cost worlds.**

| Across all 28 worlds | Result |
|---|---|
| 3-band policy beats the single cost-optimal threshold | **all 28** |
| 3-band policy beats doing nothing | **all 28** |
| 3-band structure survives (review band never collapses) | **all 28** |
| The *as-shipped* policy also beats both baselines | **all 28** |
| Single threshold ranges over | 0.1016 – 0.2034 |
| Band lo / band hi range over | 0.0346–0.0804 / 0.3167–0.6167 |

Worst-case regret from having shipped a policy tuned to the wrong costs: **₹745
per 1,000 COD orders**, in the reviewer-accuracy-70% world — about 0.5% of the
₹140,804 of loss the detector avoids on the validation fold.

**Verdict: the conclusion holds across the whole range.** No ±50% perturbation of
any economic constant flips the ordering of the three policies.

**Where it does break**, precisely: **review capacity below 2.13%**. The shipped
policy sends 2.13% of volume to humans, so a 2% cap makes it *infeasible*, not
merely suboptimal. Above ~3% the constraint is completely slack — the 3%, 5%, 7%
and 10% rows are identical, because the cost-optimal policy only wants ~2.1% in
review anyway. **The 5% constraint in the shipped config never actually binds.**
The softer spot is reviewer accuracy at or below 75%, where the optimiser shrinks
the review band to 0.82% of volume and the shipped policy's regret peaks.

## 6. Known limitations

Full list in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). The three that matter:

1. **R4-style rings that share no identifier are near-undetectable at checkout**
   (auto-action recall 0.375; a further 0.494 is routed to human review). This is
   an information limit, not a modelling gap.
2. **Address matching is exact-canonical**, so the 0.977 recall on R2 is capped
   by a synonym dictionary that knows the same vocabulary my fuzzer uses. Real
   Indian addresses include landmark directions, transliteration variance and
   missing pincodes that it will not catch. Expect it to be materially lower on
   real data.
3. **Temporal robustness looked far worse than it is.** The reported collapse
   (precision 0.9316 -> 0.3994) is mostly a generator artefact; with
   exposure-adjusted signups it recovers to 0.9197. About 2 points of PR-AUC of
   genuine six-month degradation remain. I reported the inflated version first
   and corrected it — both are above.

## 7. How to run it

```bash
pip install -r requirements.txt
```

```bash
python run_all.py
```

2-4 minutes on a laptop, from nothing to metrics (measured: 120s warm, 220s cold). Generates the data, freezes and
hashes the test set, builds features, **proves causality before training**,
trains, selects thresholds on validation, then evaluates once.

```bash
streamlit run app/streamlit_app.py
```

Batch view, review queue, and a per-order explanation panel with TreeSHAP
evidence, counter-evidence and caveats.

```bash
python run_all.py --quick
```

10× smaller dataset for a smoke test. Its metrics are **not** the reported
metrics.

The LLM explanation layer uses the Anthropic API when `ANTHROPIC_API_KEY` is
set, and falls back to a deterministic template writer otherwise, so the whole
pipeline runs on a clean machine with no key and no network.

## 8. Repo map

| Path | What |
|---|---|
| `src/generate_data.py` | Phase 1 — synthetic generator, splits, test-set freeze + hash |
| `src/features.py` | Phase 2 — point-in-time feature replay |
| `src/leakage_audit.py` | Phase 2b — causality proof + audit table |
| `src/train.py` | Phase 3 — LightGBM + isotonic calibration |
| `src/costs.py`, `src/threshold.py` | Phase 4 — rupee cost model, band selection |
| `src/explain.py` | Phase 5 — TreeSHAP attribution, LLM writer |
| `src/evaluate.py` | Phase 5 — the single frozen-test evaluation |
| `app/streamlit_app.py` | Phase 6 — review UI |
| `src/cost_sensitivity.py` | ±50% sweep of every cost constant, on validation |
| `src/diagnostic_flat_signup.py` | Generator diagnostic, writes to `data_flat/` |
| `docs/cost_model.md` | Cost constants re-derived against published rate cards |
| `docs/cost_sensitivity.md` | The 28-world sensitivity table and verdict |
| `docs/leakage_audit.md` | Per-feature placement-time audit table |
| `LOG.md` | What broke, in the order it broke |

Everything is seeded from `src/config.py`. Every cost constant carries a one-line
defence in the same file.
