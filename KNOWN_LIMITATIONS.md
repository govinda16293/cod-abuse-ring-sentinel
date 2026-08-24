# Known limitations

What this system misses, why it misses it, and what real production data would
change. Written to be read out loud in front of someone who does this for a
living.

---

## 1. What it actually misses

### R4-style rings that share no identifier — recall 0.375

The worst number in the repo, and the one I would raise first.

R4_burst_value rings share no device, no address, no phone block. Every member is
a fresh account placing 1–3 above-average COD orders in a 3–10 day window. At
checkout, the only evidence available is: account is 12 days old, payment is COD,
cart is above the category average.

That is also an exact description of a legitimate first-time COD customer. 253
R4 abuse orders in the test set: 37.5% auto-actioned, 49.4% routed to review,
13.0% missed entirely.

**This is an information limit, not a modelling gap.** No feature engineering
over an order log separates these two populations, because the distinguishing
evidence does not exist in an order log. It would take session telemetry, device
fingerprinting, delivery-door geolocation, or cross-merchant identity — none of
which a single merchant's order table contains.

The honest framing: for R4-shaped abuse this system is a **triage tool that
routes half the cases to a human**, not a detector.

### The precision cliff between 80% and 90% recall

Precision is 0.939 at 80% recall and 0.622 at 90%. Pushing past ~80% recall means
accepting roughly one false positive for every two true positives. The cost model
says do not — which is why the selected operating point sits at 80.7%.

### Prepaid abuse is entirely out of scope

405 abuse orders in the dataset are prepaid and are never scored, because the
system's only actions (force prepay, hold for review) are meaningless on an order
that is already prepaid. That is unaddressed loss, not solved loss.

### Label noise sets a ceiling

4.65% of positive labels are wrong by construction. The same model against
noiseless labels scores PR-AUC 0.941 vs 0.881 observed. About 6 points of PR-AUC
is my noise injector. Real merchant labels are noisier than 4.65% and noisier in
ways that correlate with the features — ops flags the joint family *because* it
shares an address — which would bite harder than a random flip.

---

## 2. Where the evaluation is weaker than it looks

### Temporal robustness is poor, and the review queue breaks its constraint

| | Ring-grouped | Temporal (months 13–18) |
|---|---|---|
| PR-AUC | 0.8809 | 0.6712 |
| Precision | 0.9316 | 0.3994 |
| Review share | 2.32% | **9.72%** |

The 5% manual-review capacity constraint was enforced during band selection and
is then **violated on the temporal holdout**. A system that quietly doubles its
own queue when the population shifts is not deployable as-is.

Diagnosis (measured, not assumed): 99.3% of temporal false positives come from
customers unseen in training, median tenure 12.6 days, median component size 1.00
— no identity links at all. The model learned the R4 signature and applies it to
a larger population of legitimate new accounts.

### That gap is partly my generator's fault

Signups are spread across the 18-month window, so the share of brand-new accounts
rises with calendar time — COD volume goes from 8.7k in month 1 to 27.6k in month
18, and the observed abuse rate falls from 3.8% to 1.6%. A real merchant with a
stable new-account rate would show a smaller gap. **I would not quote the 2.3x
precision drop as a production forecast.** The direction is real; the magnitude
is inflated by how I built the data.

### The test set was scored three times

Once pre-improvement, once post-improvement, and once after a unit test caught a
bug in address normalisation. Only the last run is reported. No threshold, band,
feature set or hyperparameter was ever selected using the test set — the feature
set was chosen on an out-of-period fold ending at month 12. But it is three
looks, not one, and I am not going to pretend otherwise.

### The improvement pass did not do what it was supposed to

The out-of-period selection protocol (fit 1–8, calibrate 9, select 10–12) chose
the stable feature set on expected cost. In the pre-bugfix run where I could
compare both on months 13–18, it did not beat the feature set it replaced:
temporal PR-AUC rose, temporal precision fell slightly. A selection fold with a
1–3 month gap does not contain the shift that appears at month 13. The lesson is
about validation design and I have not fixed it — doing so properly needs a
window longer than 18 months.

---

## 3. Where the synthetic data flatters the model

This is the section a panel should push hardest on, so it is the longest.

### The address matcher and the address fuzzer know each other

The generator produces spelling variants (`Flat 4B` / `Flt 4-B` / `#4B`,
`Apartments` / `Apts`, separator and case noise). The feature layer normalises
them with a synonym dictionary. The dictionary is deliberately wider than the
fuzz vocabulary, but they are still drawn from the same conceptual space, and
that is why R2 recall is as high as it is.

Real Indian addresses are far worse: landmark directions ("opp. Bata showroom,
behind the temple"), transliteration variance (Nivas / Niwas / Nivaas), missing
or wrong pincodes, building names that residents and couriers disagree about.
R2 recall here is 0.977. **Expect it to be materially lower on real data** — this
number is closer to a measure of my normaliser matching my fuzzer than of the
problem being easy. The exact-key
matcher is also brittle in the other direction — one unmatched token splits a
cluster and the graph edge silently disappears.

### Hard negatives are hard, but they are *my* hard

H1 joint families share an address and a device and get fuzzed variants; H3
returns 45–65% of a fashion basket; H2 concentrates 15–40 residents in one
building. Those cover the failure modes I thought of. Real false positives will
include shapes I did not think of: corporate/bulk buyers, resellers running a
legitimate business from a residential address, shared delivery desks in office
buildings, families that genuinely do co-ordinate their ordering.

The zero false positives on H3 and H2 should be read as "the model is not using
return rate or pincode density as a lazy proxy", **not** as "this system never
misfires on high-return customers".

### Behavioural separation is cleaner than reality

Ring campaign orders return at 86% against a blended 11% baseline. Real abuse
sits closer to legitimate behaviour and adapts. Nothing here models an adversary
who responds to being detected — the rings are static, drawn once, and never
change tactics in response to the detector. A real ring does.

### Everything is one merchant, one catalogue, five categories

No seasonality, no sales events, no cross-merchant identity, no marketplace of
sellers with different return policies. Diwali alone would break several of these
features.

### Component statistics assume the graph is trustworthy

`comp_size` treats every edge as equally real. A single bad address normalisation
or a shared public-WiFi device fingerprint merges two unrelated components, and
every member inherits the risk. There is no edge-confidence weighting and no
mechanism to split a component that was wrongly merged.

---

## 4. Things I know are wrong and left alone

- **Reviewer accuracy is a constant (88%).** Real reviewers are better on some
  ring types than others, and their accuracy is correlated with the same features
  the model uses.
- **No feedback loop.** Auto-blocked orders never produce a return outcome, so in
  production the training data would progressively stop containing the cases the
  model is most confident about. That needs a deliberate exploration hold-out and
  there is none here.
- **Isotonic calibration is fit once and never refreshed.** The temporal result
  is largely a stale-calibration result.
- **Cost constants are my estimates.** `product_loss_fraction` (0.72),
  `churn_prob_given_blocked` (0.22) and `customer_residual_value_inr` (₹1,450)
  move the threshold more than any hyperparameter. They should come from a
  merchant's finance team. The cost *framework* is the contribution; the
  constants are placeholders.
- **`comp_growth_d7` is the single dominant feature** by gain. A model leaning
  that hard on one signal is fragile to that signal being gamed or degraded.
- **The ring/household group key uses ground truth to define the split.** That is
  correct for honest measurement, but it means the split quality depends on
  knowing the rings — which in production you do not. Real ring-grouped
  evaluation needs a clustering step first, and that clustering will be wrong.

---

## 5. What would change with real production data

Ordered by how much I expect them to move the numbers:

1. **Delivery-point resolution instead of string matching** — replaces the most
   brittle component in the R2 path.
2. **Trailing-window recalibration + per-segment monitoring** — directly targets
   the temporal collapse.
3. **Session and device telemetry** — the only realistic route to R4-shaped
   rings; nothing in an order log solves them.
4. **Real ops labels with their real biases** — would lower every number here and
   is the single largest source of unknown.
5. **An adversary that adapts** — turns a static evaluation into a moving one and
   makes recall a time series rather than a scalar.
