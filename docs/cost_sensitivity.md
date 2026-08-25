# Cost sensitivity analysis

Every row re-runs threshold and band selection from scratch under a different
cost assumption, on the **validation fold**. The frozen test set is not opened
by this analysis.

Derivation and sources for the constants themselves: [cost_model.md](cost_model.md).

`re-selected` is the policy you would choose knowing the true costs.
`as-shipped` prices the policy actually shipped (lo=0.0343, hi=0.3441) in that world.
The gap is the regret from having guessed wrong, which is the number that
matters — you do not get to re-tune after the fact.

> The sweep uses a coarser 40x40 band grid than the shipped selection's 60x60,
> so the baseline row's bands (0.0346 / 0.3500) sit within one grid step of the
> shipped ones. That is grid resolution, not disagreement.

## Economic constants, swept +/-25% and +/-50%

| Constant | Setting | FN @₹2k | FP @₹2k | Single thr | Band lo | Band hi | Review % | ₹/1k re-selected | ₹/1k as-shipped | Beats single? | Beats nothing? | 3-band survives? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | baseline | 1640 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9198 | 9198 | yes | yes | yes |
| `fn_fixed` | -50% | 1540 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 8937 | 8937 | yes | yes | yes |
| `fn_fixed` | -25% | 1590 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9067 | 9067 | yes | yes | yes |
| `fn_fixed` | +25% | 1690 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9329 | 9329 | yes | yes | yes |
| `fn_fixed` | +50% | 1740 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9460 | 9460 | yes | yes | yes |
| `fn_variable` | -50% | 920 | 441 | 0.2034 | 0.0498 | 0.6167 | 1.43% | 6556 | 6633 | yes | yes | yes |
| `fn_variable` | -25% | 1280 | 441 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 7916 | 7916 | yes | yes | yes |
| `fn_variable` | +25% | 2000 | 441 | 0.1016 | 0.0346 | 0.3500 | 2.13% | 10481 | 10481 | yes | yes | yes |
| `fn_variable` | +50% | 2200 | 441 | 0.1016 | 0.0346 | 0.3500 | 2.13% | 11194 | 11194 | yes | yes | yes |
| `fp_fixed` | -50% | 1640 | 282 | 0.1016 | 0.0346 | 0.3500 | 2.13% | 8550 | 8550 | yes | yes | yes |
| `fp_fixed` | -25% | 1640 | 362 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 8874 | 8874 | yes | yes | yes |
| `fp_fixed` | +25% | 1640 | 521 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9523 | 9523 | yes | yes | yes |
| `fp_fixed` | +50% | 1640 | 601 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9847 | 9847 | yes | yes | yes |
| `fp_variable` | -50% | 1640 | 380 | 0.1016 | 0.0346 | 0.3500 | 2.13% | 8415 | 8415 | yes | yes | yes |
| `fp_variable` | -25% | 1640 | 411 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 8807 | 8807 | yes | yes | yes |
| `fp_variable` | +25% | 1640 | 472 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9590 | 9590 | yes | yes | yes |
| `fp_variable` | +50% | 1640 | 503 | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9981 | 9981 | yes | yes | yes |

## Reviewer accuracy (70–95%) and review capacity (2–10%)

| Constant | Setting | Single thr | Band lo | Band hi | Review % | ₹/1k re-selected | ₹/1k as-shipped | Beats single? | Beats nothing? | 3-band survives? |
|---|---|---|---|---|---|---|---|---|---|---|
| `review_precision` | 70% | 0.2034 | 0.0804 | 0.3167 | 0.82% | 12558 | 13303 | yes | yes | yes |
| `review_precision` | 75% | 0.2034 | 0.0804 | 0.3167 | 0.82% | 12007 | 12163 | yes | yes | yes |
| `review_precision` | 80% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 11023 | 11023 | yes | yes | yes |
| `review_precision` | 85% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9883 | 9883 | yes | yes | yes |
| `review_precision` | 90% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 8742 | 8742 | yes | yes | yes |
| `review_precision` | 95% | 0.2034 | 0.0346 | 0.6167 | 2.19% | 7483 | 7602 | yes | yes | yes |
| `max_review_share` | 2% | 0.2034 | 0.0498 | 0.3500 | 1.37% | 9855 | 9198 | yes | yes | yes |
| `max_review_share` | 3% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9198 | 9198 | yes | yes | yes |
| `max_review_share` | 5% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9198 | 9198 | yes | yes | yes |
| `max_review_share` | 7% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9198 | 9198 | yes | yes | yes |
| `max_review_share` | 10% | 0.2034 | 0.0346 | 0.3500 | 2.13% | 9198 | 9198 | yes | yes | yes |

## Verdict

Across all **28** cost worlds:

- The three-band policy beats the single cost-optimal threshold in **all** of them.
- It beats doing nothing in **all** of them.
- The three-band structure survives in **all** — the review band never collapses.
- The **as-shipped** policy also beats both baselines in every world.

How far the chosen operating point moves:

| | min | max |
|---|---|---|
| single threshold | 0.1016 | 0.2034 |
| band lo | 0.0346 | 0.0804 |
| band hi | 0.3167 | 0.6167 |

Worst-case regret from having shipped the wrong policy: **₹745 per 1,000 COD
orders**, in the `review_precision 70%` world — against a do-nothing baseline of ₹140804. So even
the worst mis-specification costs about 0.5% of the loss the detector avoids.

**Conclusion: the result holds across the whole range.** None of the four
economic constants, at ±50%, flips the ordering of the three policies or
collapses the band structure. The conclusion does not depend on my having
guessed the constants correctly — which is the point, because five of them
have no published figure at all.

## The one place it does break

**Review capacity below 2.13%.** The shipped policy sends 2.13% of validation
volume to humans. At a 2% cap it is therefore *infeasible* — not merely
suboptimal — and the optimiser is forced to a different, more expensive policy
(₹9855 vs ₹9198 re-selected). Above ~3% the constraint is completely slack:
the 3%, 5%, 7% and 10% rows are identical, because the cost-optimal policy
only wants ~2.1% of volume in review anyway. The 5% constraint in the shipped
config never actually binds.

The other soft spot is reviewer accuracy at or below 75%: the optimiser
responds by shrinking the review band sharply (review share drops to 0.82%),
and the shipped policy's regret rises to its maximum. The policy still beats
both baselines, but a review team materially worse than ~80% accurate should
re-select its bands rather than inherit these.
