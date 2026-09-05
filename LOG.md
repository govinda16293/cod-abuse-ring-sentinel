# Running log

What broke, what I thought was wrong, what was actually wrong, what I changed.
Appended as it happened, not reconstructed.

---

### 2026-08-24 - shap would not import

`pip install shap==0.46.0` succeeded but `import shap` died with
`Converting np.inexact or np.floating to a dtype not allowed`. Thought it was a
bad wheel. It was numpy: the box has numpy 2.4.6 and shap 0.46 predates the
numpy-2 dtype changes. Options were pin numpy back to 2.1 or move shap forward.
Moved shap forward to 0.48.0, which imports clean against numpy 2.4.
`requirements.txt` pins both together, because the pair matters, not either one.

### 2026-08-24 - label noise nearly destroyed the positive class

First generator run applied label noise as a flat rate over each class:
1.5% of positives flipped to 0, 0.4% of negatives flipped to 1. At a 3% base
rate, 0.4% of negatives is 115 injected false positives against 410 true
positives - 22% of all positive labels were noise. Any precision number would
have been measuring my noise injector, not the model.

Rewrote noise to be expressed relative to the positive class: 3.0% of true-abuse
orders get recorded as legit, and injected false positives equal 5.0% of the
positive count. 70% of those are drawn from hard-negative households, because
those are the cases a real ops team actually gets wrong. Noise is applied only
to COD orders, since those are the only ones a returns-abuse team adjudicates.
Realised: 217 positives hidden, 373 false positives injected, 4.65% of positive
labels are wrong.

### 2026-08-24 - the temporal split threw away 3005 groups

First cut of the temporal split dropped any group with orders on both sides of
the month-12 boundary, to honour "rings still non-crossing". That is right for
rings and catastrophic for everyone else: almost every ordinary customer spans
the boundary, so the temporal test set collapsed to accounts whose first ever
order fell after 1 Jan 2026. That is a new-account-heavy population and would
have made the temporal number incomparable to the ring-grouped one for reasons
that have nothing to do with time.

Narrowed the rule to rings only: a ring with campaign orders before the boundary
is dropped from the temporal test set (63 rings, 813 orders). Ordinary customers
and hard-negative households keep their natural behaviour, which is also the
real production setting - on 1 Jan you do already know your existing customers.

### 2026-08-24 - greedy address matching was itself a leak

Wrote the address-fuzz matcher as greedy Jaccard clustering inside a
(pincode, flat) block. Then realised it cannot pass a causality check: the
clustering is order-dependent, so an address first seen in month 15 can become
the representative that bridges two month-3 variants, and the month-3 feature
row silently moves. Exactly the class of bug this project is supposed to be
about.

Replaced it with a deterministic canonical key - (pincode, flat token, canonical
token set) - which is a pure function of a single address string and therefore
trivially causal. Cost: it is more brittle than fuzzy matching against real
address noise. Recorded in KNOWN_LIMITATIONS.md rather than papered over.

### 2026-08-24 - hard negatives were not hard

Only ring members were getting fuzzed address variants, so `addr_variant_count`
("how many spellings has this door had") separated R2 rings from legitimate
joint families almost perfectly. That signal does not exist outside my
generator: real families type their own address differently every single time.

Fixed in the generator, not the model. 60% of joint-family members now get a
fuzzed variant of the shared flat, and half of all ordinary customers' secondary
addresses are a re-typing of their own address rather than a different place.
Expect R2 recall to drop when the model is trained; that drop is the honest
number.

### 2026-08-24 - causality proof passes

`assert_prefix_invariance` truncates the world at 2026-01-01, recomputes every
feature from the truncated world alone, and requires bit-identical output
against the full run. All 55 features identical across 272,959 pre-cutoff
orders. This runs before any model is trained.

### 2026-08-24 - v1 metrics in, and the two splits disagree badly

Ring-grouped held-out test: PR-AUC 0.875, precision 0.909, recall 0.809.
Temporal held-out test (months 13-18): PR-AUC 0.611, precision 0.365, recall
0.754. Precision falls by a factor of 2.5.

First guess was that I was applying the primary model's thresholds to the
temporal model, which would confound scale differences with real degradation.
Fixed that - each model now selects its own bands on its own validation fold.
Precision moved 0.311 -> 0.365. So threshold transfer was a real bug but a small
part of the gap.

Backed out the false-positive rate from precision and base rate to see what
actually moved: FPR on legit COD orders goes 0.27% (ring-grouped) to 2.10%
(temporal). Eight times worse. The base rate also halves across the window
(3.8% in months 1-12, 1.6% in 13-18), but that alone does not explain an 8x FPR.

Ran a drift diagnostic on the LEGIT population only, comparing feature means in
months 1-12 against months 13-18. The cumulative as-of counters have all
inflated: pin_prior_orders +184%, dev_age_days +123%, comp_prior_orders +119%,
cust_prior_orders +111%, addr_prior_orders +119%. Of course they have - they
only ever go up. A model trained on months 1-12 learned that a component with
4 prior orders is unusual; by month 15 that is the median legitimate customer,
so it flags them.

So the divergence is not overfitting and not a leak. It is that roughly a third
of the feature set is a monotonically-growing counter, evaluated against a
threshold learned at a different point on that growth curve. This is the
weakest part of the build and is the thing I am going back to fix.

Also worth stating plainly: the abuse rate falls from 3.8% to 1.6% across the
window because benign order volume ramps (8.7k COD orders in month 1, 27.6k in
month 18, since customer signups are spread over time and accounts accumulate)
while ring campaign starts are uniform. That ramp is a property of my generator,
not a claim about real merchants.

### 2026-08-24 - improvement pass: added scale-free features, selected honestly

Added 8 rate-normalised features (orders per day per customer / device / address
/ cluster member, and pincode + component volume as a SHARE of marketplace
volume so far) so the same signals exist in a form that does not inflate as the
dataset ages. Causality proof re-run: still bit-identical, 63 features.

Selecting between feature sets was the awkward part. The drift I wanted to fix
lives in months 13-18, which are test. The temporal model's own validation fold
is carved out of months 1-12, so it is in-period and shows no drift - selecting
on it would have measured nothing. Built an inner protocol that uses only data
up to month 12: fit on months 1-8, calibrate on month 9, select on months 10-12,
rings non-crossing at every boundary.

Three variants on that out-of-period fold:
  A pre-improvement  (55 feats)  cost/1000 = 19,048  PR-AUC 0.855
  B plus rate feats  (63 feats)  cost/1000 = 18,440  PR-AUC 0.854
  C stable only      (49 feats)  cost/1000 = 14,441  PR-AUC 0.856
C won on expected cost, so C is what the pipeline ships.

### 2026-08-24 - the improvement did not work, and that is the finding

Retrained on C and ran the frozen test set. It did not fix the temporal gap:

                    ring-grouped            temporal
  A (v1)   PR-AUC 0.875 prec 0.909   PR-AUC 0.611 prec 0.365
  C (v2)   PR-AUC 0.872 prec 0.915   PR-AUC 0.639 prec 0.336

Temporal PR-AUC improved 0.611 -> 0.639, temporal precision got slightly WORSE,
0.365 -> 0.336. The out-of-period selection fold said C would be cheaper; on
months 13-18 it was not (29,689 vs 28,021 per 1000). The selection fold has only
a 1-3 month gap and does not contain the shift that shows up at month 13+.

I am keeping C. Switching back to A now that I have seen both on test would be
selecting on the test set, which is the thing this whole repo is supposed to not
do. Both sets of numbers are in the README.

Then I went and found out what the false positives actually ARE, instead of
theorising. On the temporal test set:

  99.3% of false-positive orders come from customers never seen in training
        (against 50.5% for legit test orders generally)
  median tenure of a false positive: 12.6 days   (legit test median: 331.6 days)
  median comp_size of a false positive: 1.00     - no identity links at all
  median comp_growth_d7 of a false positive: 0.000
  legit-order score q99 moves 0.068 (months 1-12) -> 0.694 (months 13-18)

So the diagnosis I wrote yesterday was wrong. It is not mainly counter inflation.
The false positives are brand-new accounts with no graph edges whatsoever. The
model learned in months 1-12 that "new account + COD + above-average value" is
abuse, because that is exactly the R4_burst_value signature and R4 shares no
identifier with anything by construction. In months 13-18 my generator delivers
a much larger population of legitimate brand-new accounts (signups are spread
across the window, so the new-account share rises with time), and the model
flags them.

That also explains the R4 recall of 0.41, the worst of the five signatures: a
first-time COD buyer placing one above-average order is, on the data available at
checkout, not distinguishable from an R4 member. There is no graph evidence to
find. The model is not failing to learn something; the information is not there.

Two consequences I am writing into the docs rather than engineering around:
1. The new-account ramp is a property of my generator, not a claim about real
   merchants. A real merchant with a stable new-account rate would see a smaller
   gap than the one I am reporting.
2. The deployable answer is not a cleverer feature. It is recalibrating on a
   trailing window and monitoring the flag rate on new accounts as its own
   segment. I cannot demonstrate that here without fitting on test data.

### 2026-08-24 - a unit test found a real bug in address normalisation

Wrote property tests for the pieces that carry the load. The address test failed
immediately: four spellings of one flat produced two keys, not one.

Cause: `apt` is in BOTH the filler list ("Apt 4B", where it is a prefix to throw
away) and the synonym map ("Sai Apts" -> apartments, where it is content). The
code expanded synonyms first, so the prefix `apt` in "Apt 4B, Sai Residency"
became the content token `apartments` and that address landed in a different
cluster from "Flat 4B, Sai Residency". One physical door, two clusters, graph
edge silently gone.

Fix: strip filler on the raw token first, expand synonyms on what survives.

Impact was not cosmetic. "Apt" is one of five prefix variants the fuzzer emits,
so roughly a fifth of fuzzed ring addresses were being split off. After the fix,
R2_address_fuzz recall went 0.918 -> 0.977 and overall precision 0.9153 -> 0.9316.

This is the third time the frozen test set has been scored. I am not going to
hide that: pre-improvement, post-improvement, and post-bugfix. Only the last is
reported in the README. Nothing was ever selected on it - the feature set was
chosen on an out-of-period fold ending at month 12, and the bands come from the
validation fold. The pre-bugfix A-vs-C comparison on test is superseded and I am
not quoting those numbers anywhere.

Worth saying plainly: I only found this because I wrote a test for the normaliser
in isolation. It was invisible end-to-end - the pipeline ran clean, the causality
proof passed, and the metrics looked good. A silently-worse number is the kind of
bug that survives all the way to a panel.

### 2026-08-24 - final clean run

Deleted data/ and artifacts/ entirely and ran `python run_all.py` from scratch:
119.5 seconds, and the frozen test-set hash came back identical
(06a9b5c92e4d16d6...), which is the check that the generator is genuinely
deterministic from the seed rather than accidentally reproducible.

Final: ring-grouped PR-AUC 0.8809, precision 0.9316, recall 0.8072, cost 11,617
per 1000 COD orders against 155,401 for no detector. Temporal PR-AUC 0.6712,
precision 0.3994. The gap between those two is the honest headline.

### 2026-08-25 - re-derived the cost constants against published rate cards

The whole headline rests on four numbers I asserted. Went and looked them up.

Sourced: forward shipping (Delhivery zone B-C Rs 75-150/kg; Shiprocket blended
avg shipment Rs 36-45), RTO billed at 80-100% of forward freight, COD cart
abandonment 45-55% vs prepaid 25-30%, apparel net margins 12-18% and gross
40-60%, India BPO email/chat $4-8 per agent-hour at 8-15 fraud reviews/hour.

Two of my constants came out wrong-ish and I left both alone rather than quietly
fix them, because changing a cost constant changes band selection and that means
re-scoring the frozen test set:
  * reverse shipping Rs 85 is 113% of my forward figure, above the cited 80-100%
    RTO ceiling. It overstates the cost of a miss, i.e. it biases toward
    catching more.
  * gross_margin_rate = 0.18 is really a CONTRIBUTION margin. Read as a gross
    margin it is far too low (published gross is 40-60%). The field name is
    misleading; documented rather than renamed.
Also found an omission: COD collection fees ("Rs 40 or 2%, whichever is higher")
are not in the model at all, which understates the cost of a miss and partly
offsets the reverse-shipping error.

The bigger honest finding is the split. Of eleven constants, four are sourced,
two are derived from published labour rates, and FIVE have no public figure
anywhere: product_loss_fraction, churn_prob_given_blocked,
customer_residual_value_inr, review_precision, max_review_share. Nobody
publishes what fraction of an abusive return is unrecoverable, or churn
conditional on a payment-method block. Said so in docs/cost_model.md instead of
inventing precision.

### 2026-08-25 - swept all of them; the conclusion survives

28 cost worlds: both halves of the FN and FP cost at +/-25% and +/-50%, reviewer
accuracy 70-95%, review capacity 2-10%. Each one re-runs threshold AND band
selection from scratch on validation. The test set is not opened.

The three-band policy beats the single cost-optimal threshold and beats doing
nothing in all 28, and the band structure never collapses. Single threshold only
ever moves between 0.1016 and 0.2034 - the isotonic calibrator is a step
function, so there are only a few distinct places for it to land. Worst-case
regret from having shipped a policy tuned to the wrong costs is Rs 745 per 1000
COD orders, about 0.5% of the loss the detector avoids.

One real breaking point, and it is not one of the four economic constants: a
review capacity below 2.13%. The shipped policy sends 2.13% of validation volume
to humans, so at a 2% cap it is INFEASIBLE, not merely suboptimal. The flip side
is that at 3% and above the constraint is completely slack - the 3/5/7/10% rows
are byte-identical, because the optimiser only wants ~2.1% in review anyway. So
the 5% capacity constraint I made a point of enforcing never actually binds. That
is worth knowing and I would rather say it than let someone find it.

### 2026-08-25 - the flat-signup diagnostic, and I have to correct myself

I flagged that signups ramp across the window and said the temporal precision
collapse was "real in direction, inflated in magnitude". I went and measured it
properly, and that was wrong.

The mechanism I had missed: order count per customer does not scale with how long
the customer has existed. A customer who signs up five days before the window
ends still draws the same expected number of orders, so they cram all of them
into five days. That produces, at late calendar times, a large population of
accounts that are simultaneously brand-new and ordering fast - which is precisely
the R4_burst_value signature the model keys on. My generator was manufacturing
false positives.

Regenerated ONCE with flat_signup=True (order rate scaled by each customer's
exposure fraction), into data_flat/, everything else identical. Primary frozen
test set never read or written - checked its hash before and after, unchanged.

                          ramped    flat signup   ring-grouped
  PR-AUC                  0.6712      0.8590         0.8809
  precision               0.3994      0.9197         0.9316
  recall                  0.7527      0.8167         0.8072
  review share             9.72%       0.70%          2.32%
  cost per 1000           26,294       8,841         11,617
  R4 recall               0.5101      0.6854         0.3755
  unseen-customer share    50.5%       31.2%            -
  median legit tenure     331.6d      649.6d            -
  COD month 1 -> 18    8.7k->27.6k  10.8k->16.1k        -

So temporal precision recovers to 0.9197 against a ring-grouped 0.9316, and the
review queue drops to 0.70%, comfortably inside the constraint it was blowing
through. Most of the collapse was my data, not the detector. What survives is
about 2 points of PR-AUC over six months - and since volume still ramps 1.49x
even in the flat world, even that is an upper bound.

I am not deleting the original finding. It is in the README with the correction
next to it, because "I found a scary number, then found out I had caused it" is
the actual story and the second half is the part that took the work.

Also nailed down the base-rate gap that looked unexplained: ring-grouped test is
3.18% because it samples groups across all 18 months and inherits the
dataset-wide 3.06%. Months 13-18 alone are 2.19% because benign COD volume ramps
3.18x while ring campaigns stay uniform. Removing 746 straddling-ring orders (727
of them abusive) to keep rings non-crossing takes it the rest of the way to
1.58%.

### 2026-08-25 - code frozen

No further model or feature changes. Both additions are analysis-only:
cost_sensitivity.py reads the validation fold, diagnostic_flat_signup.py writes
to a separate data_flat/ directory. The frozen test set has still been scored
three times, all before today.

### 2026-09-05 - the false-positive profile numbers were stale

While filling in the README I went to cite the temporal false-positive
diagnosis and realised those three numbers (99.3% unseen, median tenure 12.6
days, median comp_size 1.00) came from a throwaway script I ran BEFORE the
address-normalisation bugfix and the final retrain. They were never persisted to
an artifact, so nothing was checking them.

Added the profile to `diagnostic_flat_signup.py` so it writes to
`artifacts/flat_signup_diagnostic.json` under
`temporal_false_positive_profile_ramped`. Against the shipped model the real
numbers are: 2,065 false positives, 99.66% from customers unseen in training,
median tenure 10.92 days, median comp_size 1.00, median comp_growth_d7 0.0.

So 99.3% -> 99.66% and 12.6 days -> 10.92 days. The conclusion is unchanged and
slightly stronger. Corrected in README.md and KNOWN_LIMITATIONS.md. Also dropped
the "99th percentile of legit scores moves 0.068 to 0.694" claim, which came from
the same unpersisted script and which I am not going to quote without an artifact
behind it.

The earlier LOG entries keep their original numbers, because they are a record of
what I measured at the time. This entry is the correction.

### 2026-09-05 - Streamlit Cloud build failed on the pins

Deploy died in dependency install. Streamlit Community Cloud is running Python
3.14.7; `pyarrow==17.0.0` and `pandas==2.2.3` have no cp314 wheels, so uv fell
back to building from source and pyarrow's setup.py died on a missing
`pkg_resources`.

The pins are not the real problem though. The bigger one is that Cloud was being
asked to install scikit-learn, lightgbm, shap, matplotlib, scipy and anthropic,
and the deployed app imports none of them. It reads precomputed artifacts. Every
one of those was build risk for zero benefit.

Split the requirements:
  requirements.txt           app runtime only, as floors: streamlit, pandas,
                             numpy, pyarrow. This is what Cloud installs.
  requirements-pipeline.txt  the exact pins that produced every reported metric,
                             on Python 3.11.9. This is what run_all.py needs.

Checked rather than assumed two things. First, what the app can actually reach:
walked the import graph from app/streamlit_app.py through explain.py and
features.py - shap and anthropic appear, but both are lazy imports inside
functions the hosted app never enters without an API key, so omitting them does
not break the explanation panel. Second, whether pyarrow has cp314 wheels at all
before choosing a floor: pyarrow 25.0.1 lists Python 3.14 support, so `>=20`
resolves to a wheel.

Also fixed my own bug while doing it: the sed-style patch I used to prepend the
header to requirements-pipeline.txt duplicated the whole package list, so the
file briefly had twenty pins instead of ten. Caught it because the count looked
wrong.
