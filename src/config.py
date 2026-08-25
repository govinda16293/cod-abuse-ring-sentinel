"""Central configuration. Every number here is a knob a panelist can turn.

Design note: config lives in code (not YAML) so that a reviewer reading the repo
sees the parameter and its one-line rationale in the same place. No hidden state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple
import json

# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------

@dataclass
class GenConfig:
    seed: int = 20260905

    n_customers: int = 120_000
    n_orders_target: int = 500_000          # benign orders; ring orders are added on top
    start_date: str = "2025-01-01"
    months: int = 18

    # Base rate is expressed over the SCOREABLE population = COD orders.
    # The detector never scores prepaid orders, so quoting a rate over all
    # orders would flatter the problem. Both rates are recomputed and reported.
    abuse_rate_of_cod_orders: float = 0.03   # brief asks for 2-4%

    cod_share_benign: float = 0.55           # India COD-heavy e-comm assumption
    cod_share_ring: float = 0.95             # abuse needs cash-out / no prepay trail

    ring_size_min: int = 3
    ring_size_max: int = 12
    ring_size_lognorm_mu: float = 1.55       # exp(1.55) ~ 4.7 median ring size
    ring_size_lognorm_sigma: float = 0.45

    # Ring type mix. R2 (address fuzz) deliberately over-weighted: it is the
    # hard case and we want enough test-set mass to measure recall on it.
    ring_type_mix: Dict[str, float] = field(default_factory=lambda: {
        "R1_shared_device": 0.24,
        "R2_address_fuzz": 0.28,
        "R3_phone_family": 0.18,
        "R4_burst_value": 0.16,
        "R5_hybrid": 0.14,
    })

    # Hard negatives, as a fraction of the customer base.
    hard_neg_frac: float = 0.10
    hard_neg_mix: Dict[str, float] = field(default_factory=lambda: {
        "H1_joint_family": 0.34,
        "H2_hostel_pincode": 0.18,
        "H3_high_returner": 0.36,
        "H4_device_resale": 0.12,
    })

    # Label noise. Asymmetric on purpose: a merchant's historical labels miss
    # far more abuse than they invent. Flipping produces the OBSERVED label,
    # which is what we train and report on; the true label is kept in
    # ground_truth for a ceiling check only.
    #
    # Both rates are expressed RELATIVE TO THE POSITIVE CLASS. A flat rate over
    # negatives would swamp a 3% positive class: 0.4% of negatives is ~20% of
    # all positive labels, which measures label noise, not the model.
    label_noise_fn: float = 0.030          # share of true-abuse orders recorded as legit
    label_noise_fp_rel: float = 0.050      # injected false positives, as a share of positives
    label_noise_fp_from_hardneg: float = 0.70   # ...drawn mostly from hard negatives,
    #                                             because those are the cases ops gets wrong

    # Behavioural baselines
    base_return_rate: float = 0.11          # all-category blended
    ring_return_rate: float = 0.86
    address_fuzz_rate: float = 0.85         # P(a ring member's address is fuzzed)

    # Diagnostic only. See src/diagnostic_flat_signup.py and LOG.md; never set
    # for the headline dataset.
    flat_signup: bool = False

    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20

    category_return_rate: Dict[str, float] = field(default_factory=lambda: {
        "apparel": 0.22, "footwear": 0.18, "electronics": 0.07,
        "beauty": 0.06, "grocery": 0.02,
    })
    category_mix: Dict[str, float] = field(default_factory=lambda: {
        "apparel": 0.32, "footwear": 0.15, "electronics": 0.18,
        "beauty": 0.15, "grocery": 0.20,
    })
    # (median INR, lognormal sigma) per category
    category_value: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "apparel": (1250.0, 0.62), "footwear": (2100.0, 0.55),
        "electronics": (7800.0, 0.78), "beauty": (760.0, 0.58),
        "grocery": (540.0, 0.45),
    })


# ----------------------------------------------------------------------------
# Cost model (rupees). This is the differentiator; every number is defended.
# ----------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Rupee cost model.

    Each constant is tagged [SOURCED], [DERIVED] or [ASSUMED]. Full derivation,
    the published figures behind each one, and the places where a shipped value
    sits outside its sourced range are in docs/cost_model.md. Robustness across
    +/-50% on every constant is in docs/cost_sensitivity.md.

    These values were NOT changed when the derivation was done. Changing them
    changes band selection, which would require re-scoring the frozen test set.
    """

    # ---- False negative: we let an abusive COD order through ----
    # [SOURCED] Delhivery zone B-C quote Rs 75-150 per kg; Shiprocket's blended
    # "avg shipment cost" is Rs 36-45 per shipment. A <=500g parcel is ~Rs 45-55
    # base, and fuel surcharge (10-12%) plus GST (18%) lands near this figure.
    shipping_forward_inr: float = 75.0
    # [SOURCED, conservative] RTO is billed at 80-100% of forward freight. This
    # is 113% of the forward figure above, i.e. slightly ABOVE the cited ceiling,
    # so it overstates the cost of a miss. Left as-is and flagged.
    shipping_reverse_inr: float = 85.0
    # [DERIVED] No public per-order figure. India BPO email/chat is $4-8 per
    # agent-hour (~Rs 350-700/hr); a ~5 minute returns-exception touch is
    # Rs 29-58.
    handling_inr: float = 40.0
    # [ASSUMED] No public figure exists - it needs merchant-internal disposition
    # data. Not 1.0 because some abuse returns come back resaleable and some
    # claims are rejected downstream; not low, because the defining feature of
    # this abuse is that the valuable item does not come back.
    product_loss_fraction: float = 0.72
    #
    # KNOWN OMISSION: COD collection fees ("Rs 40 or 2% of order value, whichever
    # is higher") are not modelled. Including them would raise the false-negative
    # cost, so the figure above understates true loss on that axis.

    # ---- False positive: we block / force-prepay a genuine customer ----
    # [SOURCED, but read carefully] This is a CONTRIBUTION margin, not a gross
    # margin - the field name is misleading and is documented rather than
    # renamed. Indian platform gross margins are ~40-43%; apparel net margins
    # are 12-18% and electronics 8-12%.
    gross_margin_rate: float = 0.18
    # [SOURCED, wide] COD carts abandon at 45-55% vs prepaid at 25-30%, an
    # incremental ~20-25pp. Separately, D2C brands report 25-35% COD->prepaid
    # conversion, implying 65-75% do not convert. Those two readings bracket
    # 0.20-0.75; this sits in the lower-middle and is the least well-pinned
    # false-positive input.
    prepay_abandon_prob: float = 0.34
    # [ASSUMED] No public figure for churn conditional on a payment-method block.
    churn_prob_given_blocked: float = 0.22
    # [ASSUMED] No public figure for remaining gross profit per retained Indian
    # e-commerce customer at this granularity.
    customer_residual_value_inr: float = 1450.0

    # ---- Manual review ----
    # [DERIVED] India BPO Rs 350-700/hr at a cited 8-15 fraud reviews per hour
    # implies Rs 23-88 per review. This sits at the cheap/fast end.
    review_cost_inr: float = 32.0
    # [ASSUMED] Vendor-reported analyst accuracy is marketing material. Swept
    # 70-95% in the sensitivity analysis.
    review_precision: float = 0.88
    # [OPERATING CONSTRAINT, not empirical] More than this share going to humans
    # is not deployable. Swept 2-10%.
    max_review_share: float = 0.05

    # Band edges are SELECTED on validation, these are only the search bounds.
    band_search_lo: Tuple[float, float] = (0.004, 0.60)
    band_search_hi: Tuple[float, float] = (0.30, 0.95)


@dataclass
class ModelConfig:
    seed: int = 20260905
    n_estimators: int = 700
    learning_rate: float = 0.04
    num_leaves: int = 48
    min_child_samples: int = 120
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.75
    reg_lambda: float = 5.0
    calibration_method: str = "isotonic"   # monotone, non-parametric; val set is large enough


GEN = GenConfig()
COST = CostConfig()
MODEL = ModelConfig()


def dump(path: str) -> None:
    with open(path, "w") as f:
        json.dump({"gen": asdict(GEN), "cost": asdict(COST), "model": asdict(MODEL)},
                  f, indent=2, default=str)
