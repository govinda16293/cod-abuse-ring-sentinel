# COD abuse-ring sentinel

[Live demo](https://cod-abuse-ring-sentinel-b8efmjweahf8hhlzwzmwuk.streamlit.app/) &middot; [Architecture](ARCHITECTURE.md) &middot; [Known limitations](KNOWN_LIMITATIONS.md) &middot; [Build log](LOG.md)

A defence-only detector for cash-on-delivery return abuse. It scores COD orders,
chooses its decision threshold by expected rupee cost, and emits a reviewable
explanation for every flag along with what that flag does not establish.

Razorpay AI Buildathon, Track 02 (AI Risk Manager). Every figure below comes from
a file in `artifacts/` or `data/` produced by `python run_all.py`, and is
checkable against the code.

---

## 1. The problem, and what it costs

I built this for one specific loss: a customer takes delivery of a
cash-on-delivery order, files a wrong-item return, gets cash back at the door,
and the goods that come back are worthless. A single customer doing that is just
a bad customer, and I did not go after those. I went after rings, meaning groups
of accounts running the same play together while sharing a device, a flat or a
block of phone numbers, where every account looks ordinary on its own and the
evidence only exists between them. On a ₹2,000 order a miss costs the merchant
₹1,640 and a false alarm costs ₹441.40, so I chose the decision threshold on
rupees instead of on F1.

**What a miss costs**, per abusive COD order, from [`src/config.py`](src/config.py):

| Component | Value |
|---|---|
| Forward shipping | ₹75 |
| Reverse pickup (RTO) | ₹85 |
| Ops handling | ₹40 |
| Product loss, 72% of order value | 0.72 × V |
| **False negative** | **₹200 + 0.72 V** |

**What a false alarm costs**, because blocking a real customer is not free:

| Component | Value |
|---|---|
| Margin lost if they abandon on forced prepay (34% × 18%) | 0.061 × V |
| Churn (22% never return × ₹1,450 residual value) | ₹319 |
| **False positive** | **₹319 + 0.061 V** |

On a ₹2,000 order that is **₹1,640 for a miss against ₹441.40 for a false
alarm**, a ratio of 3.72:1. F1 treats them as equal. Picking the threshold on the
cost curve moves the operating point.

Each constant is re-derived against published Indian 3PL rate cards and category
margin benchmarks in [docs/cost_model.md](docs/cost_model.md), with sources. The
five constants with no public figure are labelled as assumptions. §3 reports what
happens when they are wrong.

## 2. What I built

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

Six phases, one command. Stage-by-stage detail in
[ARCHITECTURE.md](ARCHITECTURE.md).

**The decision everything else depends on:** every feature for an order placed at
time *T* uses only data from strictly before *T*. Orders and returns are replayed
as timestamped events against a mutable state object, and a return becomes
visible only at its `return_ts`, not at its `order_ts`. There is no global
identity-graph pass followed by a split, because a global graph already contains
edges built by test-set orders.

That property is verified, not asserted. `assert_prefix_invariance` deletes every
row after a cutoff date, recomputes all features from the truncated world, and
requires bit-identical output — if a feature had read a future row, deleting that
row would move it. It runs as a gate before any model is trained:

```
[leak] PASS - all 63 features bit-identical on 272,959 pre-cutoff orders
```

63 features are computed; 49 are used by the shipped model (§6).

**Model choice.** LightGBM over as-of graph summaries, not a GNN. The identity
graph here is mostly stars and small cliques, it is already summarised into ~15
component-level features, a tree is defensible feature by feature, and TreeSHAP
gives exact per-prediction attribution. Message passing would also make the
causality guarantee harder to prove, since neighbourhood aggregation needs
time-masking at every hop.

**The LLM boundary.** The classifier decides; the language model writes. It never
sees the label or the threshold, receives only a frozen payload of TreeSHAP
evidence plus rule-generated caveats, and its output has no path back into the
decision. Without an API key a deterministic template writer produces the same
content, so the repo runs offline.

### The data

Synthetic, seeded, regenerable. No scraped, purchased or real customer data.

| | |
|---|---|
| Orders | 508,349 over 18 months (2025-01-01 to 2026-07-01) |
| Customers | 122,755 |
| **Scoreable population (COD orders)** | **248,244 (48.83% of orders)** |
| Abuse rings | 512 |
| Hard-negative customers | 9.78% of the base |

Base rates are quoted over the population the detector actually scores. Its only
actions are "force prepay" and "hold for review", neither of which means anything
on an order that is already prepaid, so prepaid orders are never scored.

| Rate | Value |
|---|---|
| **Abuse rate, COD orders (observed labels)** | **3.06%** |
| Abuse rate, COD orders (noiseless labels) | 3.00% |
| Abuse rate over all orders — the flattering denominator, not used | 1.58% |
| **Abuse rate, COD-active customers** | **2.77%** |
| Abuse orders on prepaid — out of scope, unaddressed loss | 405 |

**Label noise is non-zero by construction.** 217 true-abuse COD orders are
recorded as legitimate, and 373 false positives are injected, 70% of them drawn
from hard-negative households. Net, **4.65% of positive labels are wrong**.
Training and every headline number use the observed (noisy) labels, because that
is what a merchant would actually have. Against noiseless labels the same model
scores PR-AUC 0.9411 / precision 0.9622 / recall 0.8454, so roughly 6 points of
PR-AUC is label noise rather than model error.

**Ring signatures and hard negatives.** Five ring types differ in *which*
identifier they share; four hard-negative populations each defeat one naive rule.

| Ring | Shares | | Hard negative | Naive rule it defeats |
|---|---|---|---|---|
| R1 shared_device | 1–2 devices | | H1 joint_family | "same address AND device" |
| R2 address_fuzz | one flat, many spellings | | H2 hostel_pincode | "pincode concentration" |
| R3 phone_family | adjacent MSISDNs | | H3 high_returner | "return rate > 40%" |
| R4 burst_value | **nothing** | | H4 device_resale | "device on 2 accounts" |
| R5 hybrid | 2 mechanisms, partially | | | |

Legitimate joint families also get fuzzed address variants, and half of ordinary
customers' second addresses are a re-typing of their own. Without that, "this
door has been spelled five ways" would separate R2 from real families almost
perfectly — a signal that exists nowhere outside a generator.

## 3. Results

**Frozen test set**, `sha256 = 06a9b5c92e4d16d6…73ad98`, 101,811 orders of which
49,905 are COD and scoreable. Generated and hashed in Phase 1 before any model
existed. Thresholds are selected on validation and read back from
`artifacts/policy.json`; they are never re-tuned on test.

### Headline — ring-grouped split

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
| TP / FP / FN / TN | 1,281 / 94 / 140 / 47,231 |

Precision at fixed recall:

| Recall | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|
| Precision | 0.9715 | 0.9715 | 0.9682 | 0.9393 | 0.6224 |

### Cost table (₹ per 1,000 COD orders)

| Policy | Cost | vs. no detector |
|---|---|---|
| No detector | 155,401 | — |
| Block all COD | 460,456 | 2.96× worse |
| Single cost-optimal threshold (0.2034) | 25,926 | −83.32% |
| **Three-band policy (shipped)** | **11,617** | **−92.52%** |

Where the volume lands:

| Band | Range | Share of COD orders | Action |
|---|---|---|---|
| Pass | p < 0.0343 | 94.92% | ship as COD |
| Review | 0.0343 ≤ p < 0.3441 | 2.32% | human queue |
| Auto-action | p ≥ 0.3441 | 2.76% | force prepay |

### Sensitivity — what happens when the cost constants are wrong

Five of the eleven cost constants have no published figure anywhere, so all of
them are swept: both halves of the false-negative and false-positive cost at ±25%
and ±50%, reviewer accuracy 70–95%, and review capacity 2–10%. **28 cost worlds**,
each re-running threshold *and* band selection from scratch on the validation
fold. The frozen test set is not opened by this analysis. Full table:
[docs/cost_sensitivity.md](docs/cost_sensitivity.md).

| Across all 28 worlds | Result |
|---|---|
| **Three-band structure survives — the review band never collapses** | **28 / 28** |
| Three-band policy beats the single cost-optimal threshold | 28 / 28 |
| Three-band policy beats doing nothing | 28 / 28 |
| The *as-shipped* policy also beats both baselines | 28 / 28 |

Worst-case regret from having shipped a policy tuned to the wrong costs:
**₹744.66 per 1,000 COD orders**, in the reviewer-accuracy-70% world — **0.53% of
the ₹140,804 do-nothing baseline on the validation fold**. The operating point
itself moves within a narrow range: single threshold 0.1016–0.2034, band lo
0.0346–0.0804, band hi 0.3167–0.6167.

**Where it does break.** Review capacity below **2.13%**. The shipped policy
sends 2.13% of validation volume to humans, so a 2% cap makes it *infeasible*
rather than merely suboptimal, and the optimiser is forced to a costlier policy
(₹9,855 against ₹9,198 re-selected). The flip side is worth stating too: at 3%
and above the constraint is completely slack — the 3%, 5%, 7% and 10% rows are
identical, because the cost-optimal policy only wants ~2.1% in review anyway.
**The 5% cap that band selection enforces never actually binds.** The softer spot
is reviewer accuracy at or below 75%, where the optimiser shrinks the review band
to 0.82% of volume and regret peaks.

### Recall by ring signature

A blended recall number hides this spread, which is why it is broken out:

| Ring type | Abuse orders | Recall (auto-action) | Sent to review | Missed |
|---|---|---|---|---|
| R3_phone_family | 407 | 0.9828 | 0.0098 | 0.0074 |
| R2_address_fuzz | 429 | 0.9767 | 0.0140 | 0.0093 |
| R1_shared_device | 311 | 0.8875 | 0.0386 | 0.0740 |
| R5_hybrid | 116 | 0.7845 | 0.1466 | 0.0690 |
| R4_burst_value | 253 | 0.3755 | 0.4941 | 0.1304 |
| *(injected label noise, no ring)* | 71 | 0.0000 | 0.0282 | 0.9718 |

This inverts the expectation the brief set out — R2 address fuzzing was predicted
to be the hard case and R4 burst-value the easy one. §5 explains why. The bottom
row is a sanity check: the 71 injected label-noise positives are correctly not
chased.

### False positives by legitimate population

| Population | Legit COD orders | FP rate | Sent to review | FPs |
|---|---|---|---|---|
| H4_device_resale | 604 | 0.6623% | 3.146% | 4 |
| H1_joint_family | 1,760 | 0.2273% | 4.886% | 4 |
| ordinary customers | 42,229 | 0.2037% | 1.994% | 86 |
| H3_high_returner | 2,979 | 0.0000% | 1.074% | 0 |
| H2_hostel_pincode | 746 | 0.0000% | 1.877% | 0 |

Resold devices are 3.25× more likely to be auto-actioned than an ordinary
customer, and joint families are routed to review at 2.45× the ordinary rate, so
the hard negatives are doing their job. Genuine high-return shoppers and hostel
residents produce zero false positives, which says the model is not using "high
return rate" or "pincode concentration" as a lazy proxy. It does not say the
system never misfires on those populations.

## 4. The correction

I ran the temporal split as a second evaluation, training on months 1 to 12 and
testing on months 13 to 18. Precision on the auto-action band fell from 0.9316 to
0.3994, and the review queue grew from 2.32% to 9.72% of volume, past the 5%
capacity the bands were selected under. I looked at where the false positives
came from and found that 99.66% of the 2,065 of them were customers never seen in
training, with a median tenure of 10.92 days and a median identity-component size
of 1.00, so no graph links at all. I wrote that up as the main finding of the
project.

Then I checked my own generator, and found that order count per customer never
scaled with how long the account had existed, so a customer who signed up five
days before the window closed still drew a full allocation of orders and placed
all of them inside those five days. That manufactured the exact population the
model over-flags: brand-new accounts ordering fast with no identity links. I
regenerated once with each customer's order rate scaled by their exposure to the
window, and temporal precision came back to 0.9197 against a ring-grouped 0.9316,
with the review queue at 0.70%. Most of the collapse I reported was my own bug;
what survives is 0.0219 of PR-AUC and 0.0119 of precision over six months, and I
left the original numbers standing beside the correction because the sequence is
the point.

**As originally found.** Temporal holdout, train months 1–12, test months 13–18:

| Metric | Ring-grouped | Temporal (as reported) |
|---|---|---|
| Orders scored | 49,905 | 115,690 |
| Abuse orders | 1,587 | 1,824 |
| Base rate | 3.18% | 1.58% |
| PR-AUC | 0.8809 | **0.6712** |
| Precision (auto-action) | 0.9316 | **0.3994** |
| Recall (auto-action) | 0.8072 | 0.7527 |
| Review share | 2.32% | **9.72%** |
| ₹ / 1,000 | 11,617 | 26,294 |

The false-positive profile is persisted to
`artifacts/flat_signup_diagnostic.json` under
`temporal_false_positive_profile_ramped`: 2,065 false positives, 99.66% of them
from customers never seen in training against 50.50% for legit temporal test
orders generally, median tenure 10.92 days, median `comp_size` 1.00, median
`comp_growth_d7` 0.0.

**The flat-signup diagnostic that overturned it.** In the headline generator,
order count per customer does not scale with how long the customer has existed,
so a customer signing up five days before the window ends crams a full allocation
of orders into those five days, manufacturing exactly the "brand-new account
ordering fast" population the model over-flags. Regenerated once with
`flat_signup=True`, which scales each customer's order rate by the fraction of
the window they were present for. Same seed, rings, hard negatives, label noise,
feature code, model and band-selection procedure. Writes to `data_flat/`.

| Metric | Temporal (ramped) | Temporal (flat signup) | Ring-grouped, for reference |
|---|---|---|---|
| Orders scored | 115,690 | 93,392 | 49,905 |
| Abuse orders | 1,824 | 1,795 | 1,587 |
| Base rate | 1.58% | 1.92% | 3.18% |
| PR-AUC | 0.6712 | **0.8590** | 0.8809 |
| Precision (auto-action) | 0.3994 | **0.9197** | 0.9316 |
| Recall (auto-action) | 0.7527 | 0.8167 | 0.8072 |
| Review share | 9.72% | **0.70%** | 2.32% |
| ₹ / 1,000 | 26,294 | 8,841 | 11,617 |
| R4_burst_value recall | 0.5101 | 0.6854 | 0.3755 |
| Legit test orders from unseen customers | 50.50% | 31.17% | n/a |
| Median tenure, legit test orders | 331.6 d | 649.6 d | n/a |
| COD orders, month 1 → month 18 | 8,683 → 27,624 | 10,837 → 16,138 | n/a |

This is a generator diagnostic, not a headline result, and it changes no number
in §3. It does read `data/` for the ramped side's descriptive statistics, but it
never scores the ring-grouped frozen test set, fits nothing on it and selects no
threshold from it. `data/test_set.sha256` is byte-identical before and after.
Reproduce with `python src/diagnostic_flat_signup.py`.

**The base-rate gap, since it otherwise looks unexplained.** The ring-grouped
test samples groups uniformly across all 18 months and so inherits the
dataset-wide COD abuse rate of 3.06% (3.18% after sampling). Months 13–18 alone
run at 2.19%, because benign COD volume ramps 3.18× across the window while ring
campaign starts stay uniform. Removing 746 COD orders belonging to straddling
rings, 727 of them abusive, to keep rings non-crossing takes it the rest of the
way to 1.58%.

## 5. What survives

**R4-style rings remain an information limit, not a modelling gap.**
R4_burst_value shares no device, no address and no phone block by construction.
Every member is a fresh account placing 1–3 above-average COD orders inside a
3–10 day window, which is also an exact description of a legitimate first-time
COD customer. On the frozen test set it reaches **recall 0.3755 with a further
0.4941 routed to human review** — so of 253 R4 abuse orders, 86.96% are either
actioned or queued for a person and 13.04% pass silently. For this signature the
system is a triage tool that sends about half the cases to a human, not a
detector. The evidence needed to do better is not in an order log; it would take
session telemetry, device fingerprinting, delivery-door geolocation or
cross-merchant identity.

The same limit shows up on the flat-signup dataset — R4 recall 0.6854, still the
lowest of the five signatures — and it is the mechanism behind the temporal
result in §4 seen from the other side. A detector leaning on "new account + COD +
above-average value" is genuinely exposed to any shift in the new-account mix.
That exposure is worth monitoring per-segment in production, even though my
measurement of it was inflated.

**Residual temporal degradation: about 2 points of PR-AUC, and that is an upper
bound.** On the flat-signup dataset temporal PR-AUC is 0.8590 against a
ring-grouped 0.8809 (a gap of 0.0219), and precision 0.9197 against 0.9316 (a gap
of 0.0119). Volume still ramps 1.49× across the flat window (10,837 → 16,138 COD
orders per month), because customers signing up mid-window still contribute
partial windows — so part of that remaining 2 points is still the generator, and
the true six-month degradation is smaller than 0.0219 PR-AUC, not larger.

**The precision cliff between 80% and 90% recall.** Precision is 0.9393 at 80%
recall and 0.6224 at 90%. Pushing past ~80% means accepting roughly one false
positive for every two true positives; the cost model says do not, which is why
the selected operating point sits at recall 0.8072.

## 6. Known limitations

I went looking for these instead of waiting to be asked, and every one of them
came out of checking my own work. Each item below is either measured against an
artifact in this repo or labelled as an assumption I could not source. The four
cost-model flaws are ones I found in my own constants and then chose to leave in
place, for the reason given.

Full list in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). The items I would
raise before anyone else does:

**Four self-reported flaws in the cost model**, all left unchanged, because
altering a cost constant changes band selection and would require re-scoring the
frozen test set:

1. Reverse shipping at ₹85 is **113% of my own forward figure**, above the cited
   80–100% RTO band. It overstates the cost of a miss, biasing the detector
   toward catching more.
2. `gross_margin_rate = 0.18` is a **contribution margin, not a gross margin**.
   Published gross margins are 40–60% for established apparel brands. The field
   name is misleading; documented rather than renamed.
3. **COD collection fees are not modelled at all** ("₹40 or 2% of order value,
   whichever is higher"). This understates the cost of a miss, partly offsetting
   flaw 1.
4. `prepay_abandon_prob = 0.34` sits inside a very wide sourced bracket. The two
   published readings imply anywhere from 0.20 to 0.75. It is the least
   well-pinned false-positive input.

**Five constants have no published figure anywhere** and are stated assumptions
rather than estimates dressed up: `product_loss_fraction` (0.72),
`churn_prob_given_blocked` (0.22), `customer_residual_value_inr` (₹1,450),
`review_precision` (0.88), `max_review_share` (0.05). Nobody publishes what
fraction of an abusive return is unrecoverable, or churn conditional on a
payment-method block. The defence of the result is the sensitivity sweep in §3,
not the constants. Derivation and sources: [docs/cost_model.md](docs/cost_model.md).

**The frozen test set has been scored three times**: once pre-improvement, once
post-improvement, and once after a unit test caught a bug in address
normalisation. Only the last is reported. No threshold, band, feature set or
hyperparameter was ever selected using it. The feature set was chosen on an
out-of-period fold ending at month 12. But it is three looks, not one.

**The improvement pass did not do what it was meant to.** I added 8 scale-free
rate features to fix counter drift, then selected between feature sets on an
out-of-period protocol using only data up to month 12 (fit months 1–8, calibrate
month 9, select on months 10–12, rings non-crossing throughout):

| Variant | Features | ₹/1,000 on the selection fold | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| A pre-improvement | 55 | 15,262 | 0.8701 | 0.9253 | 0.8320 |
| B + rate features | 63 | 18,689 | 0.8728 | 0.9177 | 0.8230 |
| **C stable only (shipped)** | **49** | **14,309** | **0.8790** | **0.9367** | **0.7885** |

C won on the selection fold and ships. In the pre-bugfix run where I could
compare A and C on months 13–18, C did **not** beat A there: temporal PR-AUC rose
but temporal precision fell slightly. A selection fold with a 1–3 month gap does
not contain the shift that appears at month 13. I kept C rather than switching
back after seeing test results, because switching back would be selecting on the
test set.

Three more, briefly. R2's 0.9767 recall is closer to a measure of my address
normaliser matching my own address fuzzer than of the problem being easy. The
hard negatives cover the failure modes I thought of. They miss the ones I did not:
bulk buyers, resellers at residential addresses, shared office delivery desks.
And nothing here models an adversary that adapts once it is being detected.

## 7. How to run

```bash
pip install -r requirements-pipeline.txt
```

```bash
python run_all.py
```

Measured at 239 seconds end to end on a laptop: generate data, freeze and hash
the test set, build features, **prove causality before training**, select the
feature set on an out-of-period fold, train, choose thresholds on validation,
evaluate once on the frozen test set, then run the cost sensitivity sweep and the
flat-signup diagnostic. The core six phases alone are about 2 minutes.

```bash
streamlit run app/streamlit_app.py
```

Batch view, review queue, and a per-order panel with TreeSHAP evidence,
counter-evidence and generated caveats.

```bash
python tests/test_pipeline.py
```

Property checks: address normalisation, clustering order-independence, phone
adjacency, cost asymmetry, and artifact self-consistency.

```bash
python run_all.py --quick
```

10× smaller dataset for a smoke test. Its metrics are **not** the reported
metrics.

The LLM explanation layer uses the Anthropic API when `ANTHROPIC_API_KEY` is set
and falls back to a deterministic template writer otherwise, so the pipeline runs
on a clean machine with no key and no network.

Two requirements files, on purpose. `requirements-pipeline.txt` holds the exact
versions that produced every metric above, on Python 3.11.9. `requirements.txt`
holds only what the hosted Streamlit app needs, as version floors, because
Streamlit Community Cloud runs a newer Python than the pins support and the app
reads precomputed artifacts rather than training anything.

### Repo map

| Path | What |
|---|---|
| `src/generate_data.py` | Phase 1 — generator, splits, test-set freeze + hash |
| `src/features.py` | Phase 2 — point-in-time feature replay |
| `src/leakage_audit.py` | Phase 2b — causality proof + audit table |
| `src/experiment_featureset.py` | Feature-set selection on an out-of-period fold |
| `src/train.py` | Phase 3 — LightGBM + isotonic calibration |
| `src/costs.py`, `src/threshold.py` | Phase 4 — rupee cost model, band selection |
| `src/explain.py` | Phase 5 — TreeSHAP attribution, LLM writer |
| `src/evaluate.py` | Phase 5 — the frozen-test evaluation |
| `src/cost_sensitivity.py` | 28-world cost sweep, validation only |
| `src/diagnostic_flat_signup.py` | Generator diagnostic, writes to `data_flat/` |
| `app/streamlit_app.py` | Phase 6 — review UI |
| `docs/cost_model.md` | Cost constants vs published rate cards, with sources |
| `docs/cost_sensitivity.md` | The 28-world table and verdict |
| `docs/leakage_audit.md` | Per-feature placement-time audit |
| `requirements-pipeline.txt` | Exact pins that produced every metric, Python 3.11.9 |
| `requirements.txt` | App-only runtime, version floors, what Streamlit Cloud installs |
| `LOG.md` | What broke, dated, in the order it broke |

Everything is seeded from `src/config.py`, where each cost constant is tagged
`[SOURCED]`, `[DERIVED]` or `[ASSUMED]`.
