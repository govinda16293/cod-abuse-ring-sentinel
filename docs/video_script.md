# 5-minute pitch video — script

Target 5:00. Timings are cumulative. Numbers are read straight from
`artifacts/metrics.json`; do not round anything up on camera.

---

## 0:00–0:45 — The problem (45s)

> A customer orders four thousand rupees of shoes, cash on delivery. They take
> delivery, file a "wrong item" return, and get cash back at the door. The
> merchant paid forward shipping, reverse pickup, and handling — and what comes
> back is worthless.
>
> Once, that's a bad customer. But do it with eleven accounts sharing two devices
> and one flat, over six weeks, and it's a ring. Each account looks unremarkable
> on its own. The pattern only exists *between* them.
>
> So I built a defence-only detector for cash-on-delivery return abuse. It scores
> COD orders, picks its threshold in rupees, and explains every flag.

*Screen: one abusive order in the UI, then the review-queue tab.*

---

## 0:45–1:45 — Architecture (60s)

> Six stages. Synthetic generator, point-in-time features, a calibrated boosted
> tree, a cost policy, an explanation layer, and a UI.
>
> The decision everything hangs off is this one. Every feature for an order at
> time T uses only data from before T. Orders and returns are replayed as
> timestamped events — a return only becomes visible at its return timestamp, not
> its order timestamp. There is no global graph pass followed by a split, because
> a global graph already contains edges built by test-set orders.
>
> And I don't just claim that. This deletes every row after a cutoff date,
> recomputes all sixty-three features from the truncated world, and requires
> bit-identical output. If a feature had touched a future row, removing that row
> would move it. All sixty-three features identical across two hundred and
> seventy-three thousand orders — and this runs *before* any model is trained.
>
> I chose LightGBM over a GNN deliberately. The graph is mostly stars and small
> cliques, it's already summarised into fifteen as-of component features, and a
> tree gives me exact TreeSHAP attribution I can defend feature by feature.

*Screen: the ASCII diagram from ARCHITECTURE.md, then the `[leak] PASS` line.*

---

## 1:45–3:15 — Live batch demo (90s)

> Here's a batch of held-out orders, scored. Ninety-five percent pass straight
> through. Two point three percent go to a human queue. Two point eight percent
> get auto-actioned — forced to prepay.
>
> Let's open a flagged one. Calibrated probability, the band it fell in, and the
> evidence: this account sits in a cluster of seven linked accounts, four new
> links formed in that cluster in the past week, average account age nine days.
>
> Then the part I care about more — what this does *not* establish. The address
> matched by text normalisation, not geocoding, so a normalisation error produces
> this same signal. That's generated from the feature row by rule, not written by
> hand.
>
> And the language model writes this justification from that payload. It never
> sees the label, never sees the threshold, and its output has no path back into
> the decision. The classifier decides; the LLM writes. If I'd asked a model "is
> this fraud", I'd have a second uncalibrated classifier wearing a justification
> costume.

*Screen: batch tab → order detail → evidence → caveats → generated justification.*

---

## 3:15–4:15 — Metrics and false-positive cost (60s)

> On the frozen test set: PR-AUC point eight eight one. Precision point nine
> three two, recall point eight zero seven at the operating point. Expected
> calibration error, point zero zero one eight.
>
> But the threshold isn't picked by F1. F1 assumes a false positive and a false
> negative cost the same. They don't. On a two-thousand-rupee order, missing
> abuse costs sixteen hundred and forty rupees. Wrongly blocking a real customer
> costs four hundred and forty-one. Nearly four to one.
>
> So I sweep the threshold on expected rupee cost. No detector: a hundred and
> fifty-five thousand rupees per thousand COD orders. Switch COD off: four
> hundred and sixty thousand — three times worse. This policy: eleven thousand
> six hundred. A ninety-two point five percent reduction.
>
> Now the table that matters. Recall by ring signature — and it inverts what I
> expected. Address-fuzzing rings, ninety-eight percent. Phone-block rings, also
> ninety-eight. But burst-value rings — thirty-eight percent. Those share *no*
> identifier by construction. All that's left is "new account, COD, high value",
> which is also an exact description of a legitimate first-time customer. That's
> an information limit, not a modelling gap. Another forty-nine percent of them do
> get routed to review, so it triages rather than missing them silently — but a
> blended recall number would have hidden the whole thing.

*Screen: metrics tab, cost curve, then the recall-by-ring-type table.*

---

## 4:15–5:00 — What broke, and what's next (45s)

> Two things I'd want to be asked about.
>
> First: I ran a second evaluation on a temporal holdout — train on months one to
> twelve, test on thirteen to eighteen. Precision falls from point nine three two
> to point three nine nine, and the review queue breaks its own five percent
> capacity limit. I went and found out why instead of reporting a mystery:
> ninety-nine percent of those false positives are accounts never seen in
> training, median age twelve days, with *no* identity links at all. The model
> learned the burst-value signature and meets a bigger population of legitimate
> new accounts. Same root cause as the thirty-eight percent, seen from the other
> side.
>
> Second: I tried to fix it with scale-free features, selected on an
> out-of-period fold using only data up to month twelve. It didn't work — a
> one-to-three month gap doesn't contain the shift that shows up at month
> thirteen. I kept the change anyway, because switching back after seeing test
> results is exactly the failure this repo is built to avoid. And a unit test
> later caught a real bug in my address normaliser that was quietly costing six
> points of recall on the fuzzing rings. That's in the log too.
>
> What I'd do next: recalibrate on a trailing window, monitor the new-account
> segment's flag rate on its own, and replace address string-matching with real
> delivery-point resolution. All of it's in LOG.md, dated, as it happened.

*Screen: temporal comparison table, then LOG.md scrolling.*

---

## Delivery notes

- Do not say "roughly" or "about" for any metric. Say the number.
- The R4 inversion and the temporal collapse are the strongest content. Do not
  cut them for time — cut the architecture section instead.
- If asked "why not a GNN" in the interview, the answer is in
  `src/train.py`'s docstring, not improvised.
