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
    # ---- False negative: we let an abusive COD order through ----
    # Forward shipping is paid, reverse pickup is paid, the returned item is
    # either missing or swapped for a worthless substitute, and ops burns time.
    shipping_forward_inr: float = 75.0
    shipping_reverse_inr: float = 85.0
    handling_inr: float = 40.0
    # Fraction of order value actually lost. Not 1.0: some abuse returns come
    # back as genuinely resaleable goods, some claims get rejected downstream.
    product_loss_fraction: float = 0.72

    # ---- False positive: we block / force-prepay a genuine customer ----
    gross_margin_rate: float = 0.18         # margin forgone if the order is lost
    prepay_abandon_prob: float = 0.34       # P(genuine COD customer abandons on forced prepay)
    churn_prob_given_blocked: float = 0.22  # P(customer never returns after friction)
    customer_residual_value_inr: float = 1450.0   # remaining gross profit of a retained customer

    # ---- Manual review ----
    review_cost_inr: float = 32.0           # analyst minutes, fully loaded
    review_precision: float = 0.88          # analyst correctly resolves this share of queue
    max_review_share: float = 0.05          # operating constraint: >5% to humans is not deployable

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
