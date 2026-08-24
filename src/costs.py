"""Phase 4 - the cost model, in rupees.

Every constant is in config.CostConfig with a one-line defence. The point of
this file is that "which threshold" stops being a matter of taste. F1 implicitly
asserts that a false positive and a false negative cost the same amount, which
here is false by roughly an order of magnitude in the wrong direction: letting a
2,000 rupee abusive COD order through costs about 1,640 rupees, while wrongly
forcing a genuine 2,000 rupee customer to prepay costs about 440.

Costs are per order and depend on order value, so the optimum is not a single
number derived from the base rate - it is swept.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import COST  # noqa: E402


def fn_cost(value_inr, cost=COST):
    """We passed an abusive COD order and it came back as an abusive return.

    Forward shipping is paid, reverse pickup is paid, ops handles the case, and
    the item is either missing or substituted. product_loss_fraction is 0.72 and
    not 1.0 because some abuse returns do come back resaleable and some claims
    are rejected downstream.
    """
    v = np.asarray(value_inr, dtype=np.float64)
    return (cost.shipping_forward_inr + cost.shipping_reverse_inr + cost.handling_inr
            + cost.product_loss_fraction * v)


def fp_cost(value_inr, cost=COST):
    """We forced a genuine customer to prepay (or blocked COD for them).

    Two separate losses: some fraction abandon the cart, so the margin on this
    order is gone; and some fraction never come back, so their residual value is
    gone. These are modelled independently because a customer can complete this
    order grudgingly and still churn.
    """
    v = np.asarray(value_inr, dtype=np.float64)
    return (cost.prepay_abandon_prob * cost.gross_margin_rate * v
            + cost.churn_prob_given_blocked * cost.customer_residual_value_inr)


def review_cost(n, cost=COST):
    return cost.review_cost_inr * np.asarray(n, dtype=np.float64)


def policy_cost(y, p, value, lo, hi, cost=COST):
    """Expected rupee cost of the three-band policy.

    p <  lo          -> pass
    lo <= p < hi     -> human review queue
    p >= hi          -> auto-action (force prepay / block COD)

    A reviewer is right review_precision of the time; when wrong, the full
    false-negative or false-positive cost lands anyway on top of the review fee.
    Setting lo == hi collapses this to a single-threshold policy.
    """
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    FN = fn_cost(v, cost)
    FP = fp_cost(v, cost)

    band_pass = p < lo
    band_rev = (p >= lo) & (p < hi)
    band_act = p >= hi

    c = np.zeros(len(p), dtype=np.float64)
    c[band_pass & y] = FN[band_pass & y]
    c[band_act & ~y] = FP[band_act & ~y]
    miss = 1.0 - cost.review_precision
    c[band_rev] = cost.review_cost_inr
    c[band_rev & y] += miss * FN[band_rev & y]
    c[band_rev & ~y] += miss * FP[band_rev & ~y]

    n = len(p)
    return dict(
        total_cost=float(c.sum()),
        cost_per_1000=float(c.sum() / n * 1000.0),
        pass_share=float(band_pass.mean()),
        review_share=float(band_rev.mean()),
        action_share=float(band_act.mean()),
        n_review=int(band_rev.sum()),
        tp=int((band_act & y).sum()), fp=int((band_act & ~y).sum()),
        fn=int((band_pass & y).sum()), tn=int((band_pass & ~y).sum()),
        caught_in_review=int((band_rev & y).sum()),
    )


def do_nothing_cost(y, value, cost=COST):
    """Baseline: no detector at all. Every abusive order lands."""
    y = np.asarray(y).astype(bool)
    v = np.asarray(value, dtype=np.float64)
    total = float(fn_cost(v[y], cost).sum())
    return dict(total_cost=total, cost_per_1000=total / len(y) * 1000.0)


def block_all_cod_cost(y, value, cost=COST):
    """The other baseline a merchant actually considers: switch COD off."""
    y = np.asarray(y).astype(bool)
    v = np.asarray(value, dtype=np.float64)
    total = float(fp_cost(v[~y], cost).sum())
    return dict(total_cost=total, cost_per_1000=total / len(y) * 1000.0)
