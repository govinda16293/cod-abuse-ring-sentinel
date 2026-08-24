# Leakage audit

Every feature, and whether it could actually be computed at the moment the
order is placed in production. Anything that could not be is dropped, not
down-weighted.

## Features used

| Feature | Source | Computable at placement | Why |
|---|---|---|---|
| `value_inr` | order | **YES** | Cart total, known at checkout. |
| `log_value` | order | **YES** | Transform of value_inr. |
| `item_count` | order | **YES** | Line-item count in the cart. |
| `promo_used` | order | **YES** | Coupon applied at checkout. |
| `hour_of_day` | order | **YES** | From order_ts itself. |
| `day_of_week` | order | **YES** | From order_ts itself. |
| `is_night` | order | **YES** | From order_ts itself. |
| `cat_*` | order | **YES** | Category of the cart, 5 one-hot columns. |
| `cust_tenure_days` | customer | **YES** | order_ts minus account creation date. |
| `cust_prior_orders` | customer | **YES** | Count of that customer's orders with ts < order_ts. |
| `cust_prior_cod` | customer | **YES** | As above, restricted to COD. |
| `cust_cod_share` | customer | **YES** | Laplace-smoothed share of prior orders that were COD. |
| `cust_prior_returns` | customer | **YES** | Counts a return only once return_ts < order_ts. A return in flight is NOT counted. |
| `cust_return_rate` | customer | **YES** | Smoothed (returns+1)/(orders+8), same time rule. |
| `cust_days_since_last` | customer | **YES** | Gap to previous order. |
| `cust_orders_d7 / d30` | customer | **YES** | Exponentially decayed order counters, tau=7d/30d. |
| `cust_mean_value / cust_max_value` | customer | **YES** | Running stats over prior orders. |
| `cust_value_z` | customer | **YES** | Welford z-score of this cart vs the customer's history. |
| `cust_distinct_devices` | customer | **YES** | Distinct devices used before now. |
| `cust_distinct_addr_clusters` | customer | **YES** | Distinct address clusters shipped to before now. |
| `dev_distinct_customers` | device | **YES** | Other accounts seen on this device_id before now. Device id is present in the checkout event. |
| `dev_prior_orders` | device | **YES** | Orders from this device before now. |
| `dev_return_rate` | device | **YES** | Smoothed, settled returns only. |
| `dev_age_days` | device | **YES** | Time since the device was first seen. |
| `addr_distinct_customers` | address | **YES** | Other accounts that have shipped to this address CLUSTER before now. |
| `addr_prior_orders` | address | **YES** | Orders to this cluster before now. |
| `addr_return_rate` | address | **YES** | Smoothed, settled returns only. |
| `addr_variant_count` | address | **YES** | Distinct raw spellings of this cluster seen before now. |
| `addr_is_new_variant_of_known_cluster` | address | **YES** | This exact string is new but the cluster is known. Both facts are available at checkout. |
| `phone_neighbours_active` | phone | **YES** | Accounts on MSISDNs within +/-5 of this one that have transacted before now. |
| `phone_neighbour_orders` | phone | **YES** | Their prior order count. |
| `pin_distinct_customers` | address | **YES** | Accounts active in this pincode before now. |
| `pin_prior_orders` | address | **YES** | Orders in this pincode before now. |
| `pin_return_rate` | address | **YES** | Smoothed, settled returns only. |
| `comp_size` | identity graph | **YES** | Customers in the connected component, using only edges evidenced before now plus the device/address/phone this order itself declares. |
| `comp_prior_orders` | identity graph | **YES** | Component order count before now. |
| `comp_return_rate` | identity graph | **YES** | Smoothed, settled returns only. |
| `comp_cod_share` | identity graph | **YES** | Smoothed COD share of the component. |
| `comp_orders_d7 / d30` | identity graph | **YES** | Decayed component order counters. |
| `comp_growth_d7` | identity graph | **YES** | Decayed count of merges into this component. |
| `comp_mean_tenure_days` | identity graph | **YES** | Mean account age of members. |
| `comp_max_tenure_days` | identity graph | **YES** | Age of the oldest member account. |
| `comp_distinct_devices` | identity graph | **YES** | Devices seen inside the component. |
| `comp_distinct_addr_clusters` | identity graph | **YES** | Address clusters inside the component. |
| `comp_value_per_customer` | identity graph | **YES** | Component GMV / members. |
| `comp_orders_per_customer` | identity graph | **YES** | Component orders / members. |
| `cat_return_rate_prior` | catalogue | **YES** | Marketplace-wide return rate for the category, as-of. |
| `value_over_cat_mean` | catalogue | **YES** | Cart value over the running category mean. |

## Fields deliberately excluded

| Field | Verdict | Reason |
|---|---|---|
| `return_flag (this order)` | NO - resolves days after placement | This is the outcome. Using it is the single most common way to fake this model. |
| `return_ts / return_reason (this order)` | NO - post-placement | Only used to schedule the state update that makes the return visible to LATER orders. |
| `refund_mode / outcome` | NO - post-placement | Adjudication result, weeks later. |
| `delivered_flag` | NO - post-placement | Delivery confirmation is days after checkout. Dropped even though it correlates. |
| `is_abuse / is_abuse_true` | NO - label | Target. |
| `ring_id / ring_type` | NO - ground truth | Never a feature; lives in a separate file. |
| `household_id / is_hard_neg / hn_type` | NO - ground truth | Generator internals. |
| `addresses.building_id / flat_num / wing` | NO - generator internal | Production only has the typed address STRING. The model must re-derive the cluster from text, which is why address_cluster is computed from raw_line + pincode. |
| `phones.phone_family_id` | NO - generator internal | Production only has the number. Adjacency is re-derived numerically from the MSISDN. |
| `customers.segment / return_mult / lam_mult / cat_override` | NO - generator internal | These are the latent parameters that generated the behaviour. Using them is circular. |
| `customers.signup_ts` | YES - used | Account creation date is genuinely known at checkout. |

Enforcement, not just documentation:

* `features.build_features` refuses to run if it is handed any column outside an
  explicit allow-list (`ALLOWED_ORDER_COLS` and friends). The dropped fields
  above cannot reach the feature code by accident.
* `assert_prefix_invariance` recomputes every feature from a world truncated at
  a cutoff date and requires bit-identical output. This is run by `run_all.py`
  before any model is trained, and it raises `LeakageError` on any difference.
* Returns are replayed as separate events that fire at `return_ts`, not at
  `order_ts`. A return that has been requested but not yet settled is invisible.

