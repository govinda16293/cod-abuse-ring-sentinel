# Cost model — derivation and sources

The headline rests on four asserted constants. This document does two things:
re-derives each against published Indian 3PL rate cards and category margin
benchmarks, and states plainly which ones have **no public figure at all** so
that the reasoning is visible instead of dressed up as precision.

**The shipped constants were not changed by this exercise.** Changing them would
change band selection and therefore require re-scoring the frozen test set, and
the test set is not open. Where a sourced range disagrees with a shipped value,
that is recorded below and its effect is quantified in
[cost_sensitivity.md](cost_sensitivity.md) instead.

---

## False negative — we let an abusive COD order through

**Shipped: ₹200 + 0.72 × V**

| Constant | Shipped | Sourced range | Verdict |
|---|---|---|---|
| `shipping_forward_inr` | ₹75 | Delhivery Zone B–C ₹75–150 per **kg**; Shiprocket blended "avg shipment cost" ₹36–45 per shipment by plan tier. Add fuel surcharge 10–12% and GST 18%. | **Defensible.** A ≤500 g parcel has a base rate near ₹45–55; ₹75 is roughly that plus surcharges and GST. At the top of the plausible band for a light parcel, about right for 1 kg. |
| `shipping_reverse_inr` | ₹85 | RTO is charged at **80–100% of forward freight** (ClickPost); other operators quote 70–80%. A published Delhivery rate card shows indicative zonal RTO for ≤500 g of ₹27 / ₹32 / ₹38 / ₹41 / ₹45 for zones A–E. | **Slightly conservative.** ₹85 is 113% of my forward figure, above the cited 80–100% ceiling. This *overstates* the false-negative cost, i.e. it biases the detector toward catching more. Flagged rather than silently corrected. |
| `handling_inr` | ₹40 | **No public per-order figure exists.** Derived: India BPO email/chat support runs $4–8 per agent-hour (≈₹350–700/hr at ₹87/USD); a returns-exception touch of ~5 minutes is ₹29–58. | **In range**, but derived, not sourced. |
| `product_loss_fraction` | 0.72 | **No public figure exists.** No 3PL or merchant publishes what fraction of an abusive return's value is unrecoverable — the number would require merchant-internal disposition data. | **Assumption.** Reasoning: not 1.0 because some abuse returns come back resaleable and some claims are rejected downstream; not low, because the defining feature of this abuse is that the valuable item does not come back. Swept ±50%. |

**Known omission.** COD collection fees — "₹40 or 2% of order value, whichever
is higher" (ClickPost) — are **not** in the model. Including them would raise the
false-negative cost further. The shipped figure is therefore an underestimate of
true loss on this axis, which partly offsets the conservative reverse-shipping
number above.

## False positive — we force a genuine customer to prepay

**Shipped: ₹319 + 0.061 × V**

| Constant | Shipped | Sourced range | Verdict |
|---|---|---|---|
| `prepay_abandon_prob` | 0.34 | COD carts abandon at **45–55%**, prepaid at **25–30%** (Shiprocket Checkout) — an incremental ~20–25 pp from removing COD. Separately, D2C brands report **25–35% COD→prepaid conversion**, implying 65–75% do *not* convert. | **Mid-range but wide.** The two readings bracket 0.20–0.75; 0.34 sits in the lower-middle. The spread here is genuinely large and is the least well-pinned FP input. Swept ±50%. |
| `gross_margin_rate` | 0.18 | Indian platform gross margins ~40–43%; established apparel/accessory brands 40–60% gross. Net margins: apparel 12–18%, electronics 8–12% (Onramp benchmarks). | **The field name is misleading.** 0.18 is a *contribution* margin — after marketing, payment and fulfilment — not a gross margin. Read as gross it is far too low; read as contribution it sits at the top of the apparel net-margin range. Documented rather than renamed, because renaming the field would change nothing numerically and the shipped artefacts reference it. |
| `churn_prob_given_blocked` | 0.22 | **No public figure exists.** Merchant-internal retention data; nobody publishes churn conditional on a payment-method block. | **Assumption.** Swept via `fp_fixed` ±50%. |
| `customer_residual_value_inr` | ₹1,450 | **No public figure exists** for remaining gross profit per retained Indian e-commerce customer at this granularity. | **Assumption.** Swept via `fp_fixed` ±50%. |

## Review economics

| Constant | Shipped | Sourced range | Verdict |
|---|---|---|---|
| `review_cost_inr` | ₹32 | India BPO email/chat $4–8 per agent-hour (≈₹350–700/hr); manual fraud review throughput cited at **8–15 orders per hour**. That implies ₹23–88 per review. | **In range**, at the cheap/fast end. |
| `review_precision` | 0.88 | **No public figure exists.** Vendor-reported analyst accuracy is marketing material. | **Assumption.** Swept 70–95%. |
| `max_review_share` | 5% | Not an empirical constant — an operating constraint I imposed. | Swept 2–10%. |

---

## Net position

Of eleven constants: **four are sourced to published figures** (forward
shipping, reverse/RTO ratio, review cost, and the abandonment inputs), **two are
derived from published labour rates** (handling, review cost), and **five have no
public figure and are stated assumptions** (`product_loss_fraction`,
`churn_prob_given_blocked`, `customer_residual_value_inr`, `review_precision`,
`max_review_share`).

That is the honest split. The defence of the result is therefore not "these
constants are right" — it is the sensitivity analysis, which shows the
conclusion holds across ±50% on every one of them.

## Sources

- [Delhivery Courier Charges: Domestic & International Rates — ClickPost](https://www.clickpost.ai/blog/delhivery-courier-charges) — zone rates per kg, RTO at 80–100% of forward freight, COD fee "₹40 or 2%", fuel surcharge 10–12%, GST 18%.
- [Delhivery Rate Card — Delhivery One Help Center](https://help.delhivery.com/docs/rate-card) — zonal rate card structure.
- [E-commerce Shipping Plans & Pricing — Shiprocket](https://www.shiprocket.in/pricing/) — average shipment cost ₹36–45 by plan tier.
- [What do we mean by shipment forward charges and RTO charges? — Shiprocket Support](https://support.shiprocket.in/support/solutions/articles/43000664183-what-do-we-mean-by-shipment-forward-charges-and-rto-charges-) — RTO definition and billing basis.
- [Cart Abandonment Statistics 2025 — Shiprocket Checkout](https://checkout.shiprocket.in/blog/latest-cart-abandonment-statistics-2025/) — COD 45–55% vs prepaid 25–30% abandonment.
- [COD to Prepaid Conversion: How D2C Brands Are Getting 25–35% Conversion — OneflowAI](https://oneflowai.in/blog/cod-to-prepaid-conversion-d2c-india) — forced/incentivised COD→prepaid conversion rates.
- [Cash-on-Delivery in Indian D2C E-Commerce: Risks & Solutions — Pragma](https://www.bepragma.ai/blogs/getting-in-bed-with-cod-cash-on-delivery) — COD risk and RTO context.
- [10 Profit Margin Benchmarks for eCommerce 2025 — Onramp Funds](https://www.onrampfunds.com/resources/10-profit-margin-benchmarks-for-ecommerce-2025) — apparel 12–18% net, electronics 8–12% net, 40–60% gross for established apparel brands.
- [BPO Pricing 2026: Hourly Rates by Country & Service Type — Globalify](https://globalify.com/bpo-pricing) — India BPO $8–15/hr blended.
- [BPO Cost in India 2026 — AB7 Solutions](https://www.ab7solutions.com/pricing/bpo-cost-in-india) — $6–14/agent-hour voice, $4–8/hr email and chat.
- [Ecommerce Order Fraud Detection — US Tech Automations](https://ustechautomations.com/resources/blog/ecommerce-order-fraud-detection-pain-solution-2026) — manual fraud review throughput of 8–15 orders per hour.

Figures were read in August 2026; rate cards change and negotiated enterprise
rates are materially lower than published ones.
