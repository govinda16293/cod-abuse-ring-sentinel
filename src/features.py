"""Phase 2 - point-in-time ("as-of") feature engineering.

The whole file exists to enforce one rule: a feature for an order placed at time
T may only depend on facts that were true strictly before T.

That rules out the obvious shortcut of building one global identity graph and
then splitting, because the global graph contains edges created by test-set
orders and by returns that had not happened yet. Recall computed that way is
fiction.

Mechanism: every order and every return is turned into a timestamped event.
Events are replayed in time order against a mutable state object. For an order
event we

  1. resolve the structural links the order itself declares (device, address
     string, phone number) - these ARE known at checkout;
  2. union the customer into the identity graph using only prior activity;
  3. emit the feature row from current state;
  4. only then fold this order's own behaviour into the state.

Step 4 happening after step 3 is what makes the row causal. The property is
verified, not asserted: see leakage_audit.assert_prefix_invariance.
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DAY = 86400.0
TAU7 = 7 * DAY
TAU30 = 30 * DAY

# Columns the feature builder is allowed to see. Anything not on this list is a
# post-outcome field or a generator internal and is rejected at load time.
ALLOWED_ORDER_COLS = {"order_id", "customer_id", "device_id", "address_id", "phone_id",
                      "order_ts", "payment_mode", "category", "value_inr", "item_count",
                      "promo_used"}
ALLOWED_CUSTOMER_COLS = {"customer_id", "signup_ts"}
ALLOWED_ADDRESS_COLS = {"address_id", "raw_line", "pincode"}
ALLOWED_PHONE_COLS = {"phone_id", "msisdn"}
# return_flag/return_ts are used ONLY to build return events that fire at
# return_ts; no feature reads the current order's own return outcome.
ALLOWED_RETURN_COLS = {"order_id", "return_ts"}


# ---------------------------------------------------------------------------
# address normalisation
# ---------------------------------------------------------------------------

# Standard Indian address-component abbreviations. This is the production
# approach (a synonym dictionary), not a trick: the list is deliberately wider
# than the variants the generator actually emits. Its coverage is nevertheless
# the hard ceiling on how well R2-style address fuzzing can be caught, which is
# recorded in KNOWN_LIMITATIONS.md.
SYNONYMS = {
    "apts": "apartments", "apt": "apartments", "aprtmnt": "apartments",
    "apartment": "apartments", "appartments": "apartments",
    "res": "residency", "resi": "residency", "residancy": "residency",
    "twr": "towers", "tower": "towers", "twrs": "towers",
    "bldg": "building", "bld": "building", "blg": "building",
    "hts": "heights", "ht": "heights",
    "encl": "enclave", "cmplx": "complex", "cmpx": "complex",
    "chmbrs": "chambers", "chmbr": "chambers",
    "rd": "road", "marg": "road", "st": "street", "ln": "lane",
    "sec": "sector", "nr": "near", "opp": "opposite",
    "gr": "ground", "flr": "floor", "fl": "floor",
}
FILLER = {"flat", "flt", "apt", "no", "number", "num", "house", "h", "the", "wing",
          "block", "blk", "door", "d"}
FLAT_RE = re.compile(r"^\d{1,4}[a-h]?$")
TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalise_address(raw: str):
    """Return (flat_token, frozenset(other_tokens)).

    'Flat 4B, Sai Residency, MG Road'  and
    'Flt 4-B, SAI RES.  MG Road'       must land on the same pair.
    """
    s = TOKEN_RE.sub(" ", str(raw).lower()).strip()
    toks = s.split()
    # re-join a digit run followed by a single letter: "4 b" / "4-b" -> "4b"
    merged = []
    i = 0
    while i < len(toks):
        if (i + 1 < len(toks) and toks[i].isdigit() and len(toks[i + 1]) == 1
                and toks[i + 1].isalpha()):
            merged.append(toks[i] + toks[i + 1])
            i += 2
        else:
            merged.append(toks[i])
            i += 1
    # Filler removal MUST happen before synonym expansion. "apt" is both a
    # filler prefix ("Apt 4B") and an abbreviation of a building-name token
    # ("Sai Apts"). Expanding first turned the prefix into the content token
    # "apartments", which split one physical door into two clusters and quietly
    # cost R2 recall. Strip on the raw token, then expand what survives.
    flat = None
    rest = []
    for t in merged:
        if t in FILLER:
            continue
        t = SYNONYMS.get(t, t)
        if flat is None and FLAT_RE.match(t):
            flat = t
            continue
        rest.append(t)
    return (flat or "?"), frozenset(rest)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build_address_clusters(addresses: pd.DataFrame):
    """Cluster address strings that denote the same physical door.

    The cluster key is a pure function of one address: (pincode, flat token,
    canonical token set). That determinism is not a stylistic choice - it is
    what makes the clustering causal.

    The obvious alternative, greedy fuzzy matching inside a (pincode, flat)
    block, is order-dependent: an address first seen in month 15 can act as the
    representative that bridges two month-3 variants, so the month-3 feature row
    would silently depend on a future row. That fails prefix-invariance, and it
    is exactly the class of leak this project is supposed to avoid. A production
    fuzzy matcher would have to be rebuilt over a trailing window to stay causal.
    """
    keys = {}
    cluster = {}
    next_cid = 0
    for a, raw, pin in zip(addresses["address_id"].to_numpy(),
                           addresses["raw_line"].to_numpy(),
                           addresses["pincode"].to_numpy()):
        f, t = normalise_address(raw)
        k = (int(pin), f, t)
        cid = keys.get(k)
        if cid is None:
            cid = next_cid
            keys[k] = cid
            next_cid += 1
        cluster[int(a)] = cid
    return cluster, next_cid


def build_phone_neighbours(phones: pd.DataFrame, window: int = 5):
    """Map phone_id -> phone_ids whose MSISDN is numerically within +/- window.

    Blocks of consecutive numbers bought together are a real signal and the
    number is known at checkout, so this is a legitimate placement-time feature.
    The relation itself is static; what is time-dependent is which neighbours
    have transacted yet, which the replay handles.
    """
    nums = phones["msisdn"].astype(np.int64).to_numpy()
    pids = phones["phone_id"].to_numpy()
    order = np.argsort(nums, kind="mergesort")
    nums_s, pids_s = nums[order], pids[order]
    nb = {}
    j0 = 0
    for k in range(len(nums_s)):
        while nums_s[j0] < nums_s[k] - window:
            j0 += 1
        j1 = k
        while j1 + 1 < len(nums_s) and nums_s[j1 + 1] <= nums_s[k] + window:
            j1 += 1
        neigh = [int(p) for p in pids_s[j0:j1 + 1] if int(p) != int(pids_s[k])]
        if neigh:
            nb[int(pids_s[k])] = neigh
    return nb


# ---------------------------------------------------------------------------
# identity graph with mergeable, time-decayed component statistics
# ---------------------------------------------------------------------------

def _decay(val, last_ts, now, tau):
    if val == 0.0:
        return 0.0
    dt = now - last_ts
    if dt <= 0:
        return val
    return val * math.exp(-dt / tau)


class IdentityGraph:
    """Union-find over customers, with aggregates that survive a merge.

    Edges are created only by observed co-use of a device, an address cluster or
    an adjacent phone number. Components therefore only ever grow, which is
    exactly the as-of semantics we need: the component at time T is built from
    edges whose evidence predates T.
    """

    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)
        self.size = np.ones(n, dtype=np.int64)
        self.orders = np.zeros(n, dtype=np.int64)
        self.returns = np.zeros(n, dtype=np.int64)
        self.cod = np.zeros(n, dtype=np.int64)
        self.value = np.zeros(n, dtype=np.float64)
        self.sum_signup = np.zeros(n, dtype=np.float64)
        self.min_signup = np.full(n, np.inf, dtype=np.float64)
        self.d7 = np.zeros(n, dtype=np.float64)
        self.d30 = np.zeros(n, dtype=np.float64)
        self.d_ts = np.zeros(n, dtype=np.float64)
        self.g7 = np.zeros(n, dtype=np.float64)
        self.g_ts = np.zeros(n, dtype=np.float64)
        self.devs = defaultdict(set)
        self.addrs = defaultdict(set)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a, b, now):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        elif self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
        self.orders[ra] += self.orders[rb]
        self.returns[ra] += self.returns[rb]
        self.cod[ra] += self.cod[rb]
        self.value[ra] += self.value[rb]
        self.sum_signup[ra] += self.sum_signup[rb]
        self.min_signup[ra] = min(self.min_signup[ra], self.min_signup[rb])
        self.d7[ra] = _decay(self.d7[ra], self.d_ts[ra], now, TAU7) + \
            _decay(self.d7[rb], self.d_ts[rb], now, TAU7)
        self.d30[ra] = _decay(self.d30[ra], self.d_ts[ra], now, TAU30) + \
            _decay(self.d30[rb], self.d_ts[rb], now, TAU30)
        self.d_ts[ra] = now
        self.g7[ra] = _decay(self.g7[ra], self.g_ts[ra], now, TAU7) + \
            _decay(self.g7[rb], self.g_ts[rb], now, TAU7) + 1.0   # a merge IS growth
        self.g_ts[ra] = now
        if len(self.devs[rb]) > len(self.devs[ra]):
            self.devs[ra], self.devs[rb] = self.devs[rb], self.devs[ra]
        self.devs[ra] |= self.devs.pop(rb, set())
        if len(self.addrs[rb]) > len(self.addrs[ra]):
            self.addrs[ra], self.addrs[rb] = self.addrs[rb], self.addrs[ra]
        self.addrs[ra] |= self.addrs.pop(rb, set())
        return ra


FEATURE_COLUMNS = [
    # order-intrinsic (all known at checkout)
    "value_inr", "log_value", "item_count", "promo_used", "hour_of_day", "day_of_week",
    "is_night", "cat_apparel", "cat_footwear", "cat_electronics", "cat_beauty", "cat_grocery",
    # customer, as-of
    "cust_tenure_days", "cust_prior_orders", "cust_prior_cod", "cust_cod_share",
    "cust_prior_returns", "cust_return_rate", "cust_days_since_last", "cust_orders_d7",
    "cust_orders_d30", "cust_mean_value", "cust_value_z", "cust_max_value",
    "cust_distinct_devices", "cust_distinct_addr_clusters",
    # device, as-of
    "dev_distinct_customers", "dev_prior_orders", "dev_return_rate", "dev_age_days",
    # address cluster, as-of
    "addr_distinct_customers", "addr_prior_orders", "addr_return_rate",
    "addr_variant_count", "addr_is_new_variant_of_known_cluster",
    # phone, as-of
    "phone_neighbours_active", "phone_neighbour_orders",
    # pincode, as-of
    "pin_distinct_customers", "pin_prior_orders", "pin_return_rate",
    # identity component, as-of
    "comp_size", "comp_prior_orders", "comp_return_rate", "comp_cod_share",
    "comp_orders_d7", "comp_orders_d30", "comp_growth_d7", "comp_mean_tenure_days",
    "comp_max_tenure_days", "comp_distinct_devices", "comp_distinct_addr_clusters",
    "comp_value_per_customer", "comp_orders_per_customer",
    # category context, as-of
    "cat_return_rate_prior", "value_over_cat_mean",
    # --- scale-free rates -------------------------------------------------
    # Everything above that is a cumulative count grows monotonically with
    # calendar time, so a threshold learned in month 3 means something different
    # in month 15. These are the same signals expressed as a rate (per day, or
    # as a share of marketplace volume so far) and do not drift with the age of
    # the dataset. See LOG.md, "the two splits disagree badly".
    "cust_orders_per_day", "dev_orders_per_day", "addr_orders_per_day",
    "comp_orders_per_member_per_day", "addr_age_days",
    "pin_orders_share", "pin_cust_share", "comp_orders_share",
]

# Features whose value is a running total and therefore inflates as the dataset
# ages. Kept for the primary model, optionally dropped for drift robustness.
DRIFT_PRONE_COLUMNS = [
    "cust_prior_orders", "cust_prior_cod", "cust_prior_returns",
    "cust_mean_value", "cust_max_value",
    "dev_prior_orders", "dev_age_days",
    "addr_prior_orders", "addr_variant_count",
    "pin_prior_orders", "pin_distinct_customers",
    "comp_prior_orders", "comp_value_per_customer", "comp_orders_per_customer",
]
STABLE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in DRIFT_PRONE_COLUMNS]

CATEGORIES = ["apparel", "footwear", "electronics", "beauty", "grocery"]


def build_features(orders, customers, addresses, phones, returns, progress=True):
    """Replay the order/return stream and emit one causal feature row per order."""
    for df, allowed, name in [(orders, ALLOWED_ORDER_COLS, "orders"),
                              (customers, ALLOWED_CUSTOMER_COLS, "customers"),
                              (addresses, ALLOWED_ADDRESS_COLS, "addresses"),
                              (phones, ALLOWED_PHONE_COLS, "phones"),
                              (returns, ALLOWED_RETURN_COLS, "returns")]:
        extra = set(df.columns) - allowed
        if extra:
            raise ValueError("feature builder was handed non-placement-time columns on "
                             + name + ": " + str(sorted(extra)))

    addr_cluster, _ = build_address_clusters(addresses)
    addr_pin = dict(zip(addresses.address_id.to_numpy().tolist(),
                        addresses.pincode.to_numpy().tolist()))
    phone_nb = build_phone_neighbours(phones)

    n_cust = int(customers.customer_id.max()) + 1
    signup = np.zeros(n_cust, dtype=np.float64)
    signup[customers.customer_id.to_numpy()] = \
        customers.signup_ts.to_numpy().astype("datetime64[s]").astype(np.float64)

    o = orders.sort_values("order_id", kind="mergesort").reset_index(drop=True)
    ots = o.order_ts.to_numpy().astype("datetime64[s]").astype(np.int64)
    oid = o.order_id.to_numpy()
    cid = o.customer_id.to_numpy()
    did = o.device_id.to_numpy()
    aid = o.address_id.to_numpy()
    pid = o.phone_id.to_numpy()
    val = o.value_inr.to_numpy(dtype=np.float64)
    itemc = o.item_count.to_numpy()
    promo = o.promo_used.to_numpy()
    is_cod = (o.payment_mode.to_numpy() == "COD")
    cat = o.category.to_numpy()
    cat_ix = {c: i for i, c in enumerate(CATEGORIES)}
    cat_i = np.array([cat_ix[c] for c in cat], dtype=np.int64)

    ret_map = dict(zip(returns.order_id.to_numpy().tolist(),
                       returns.return_ts.to_numpy().astype("datetime64[s]")
                       .astype(np.int64).tolist()))

    # ---- event queue: returns settle before orders at an identical timestamp
    ev = [(int(ots[i]), 1, i) for i in range(len(o))]
    ev += [(int(t), 0, int(np.searchsorted(oid, k))) for k, t in ret_map.items()]
    ev.sort(key=lambda x: (x[0], x[1]))

    G = IdentityGraph(n_cust)
    c_orders = np.zeros(n_cust, dtype=np.int64)
    c_cod = np.zeros(n_cust, dtype=np.int64)
    c_returns = np.zeros(n_cust, dtype=np.int64)
    c_sum = np.zeros(n_cust, dtype=np.float64)
    c_m2 = np.zeros(n_cust, dtype=np.float64)
    c_max = np.zeros(n_cust, dtype=np.float64)
    c_last = np.zeros(n_cust, dtype=np.float64)
    c_d7 = np.zeros(n_cust, dtype=np.float64)
    c_d30 = np.zeros(n_cust, dtype=np.float64)
    c_dts = np.zeros(n_cust, dtype=np.float64)
    seen = np.zeros(n_cust, dtype=bool)
    c_devs = defaultdict(set)
    c_addrs = defaultdict(set)

    dev_cust = defaultdict(set)
    dev_orders = defaultdict(int)
    dev_returns = defaultdict(int)
    dev_first = {}
    ac_cust = defaultdict(set)
    ac_orders = defaultdict(int)
    ac_returns = defaultdict(int)
    ac_variants = defaultdict(set)
    ac_first = {}
    global_orders = 0
    global_customers = 0
    pin_cust = defaultdict(set)
    pin_orders = defaultdict(int)
    pin_returns = defaultdict(int)
    ph_cust = defaultdict(set)
    ph_orders = defaultdict(int)
    cat_orders = np.zeros(len(CATEGORIES), dtype=np.int64)
    cat_returns = np.zeros(len(CATEGORIES), dtype=np.int64)
    cat_value = np.zeros(len(CATEGORIES), dtype=np.float64)

    nF = len(FEATURE_COLUMNS)
    out = np.zeros((len(o), nF), dtype=np.float32)
    F = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    n_ev = len(ev)

    for e_i, (t, kind, i) in enumerate(ev):
        if kind == 0:
            # a return settles: it becomes visible to every later order
            c = int(cid[i])
            c_returns[c] += 1
            dev_returns[int(did[i])] += 1
            ac = addr_cluster[int(aid[i])]
            ac_returns[ac] += 1
            pin_returns[int(addr_pin[int(aid[i])])] += 1
            cat_returns[cat_i[i]] += 1
            root = G.find(c)
            G.returns[root] += 1
            continue

        if progress and (e_i % 100000 == 0):
            print("[feat] event %d/%d" % (e_i, n_ev), flush=True)

        c = int(cid[i])
        d = int(did[i])
        a = int(aid[i])
        ac = addr_cluster[a]
        pin = int(addr_pin[a])
        ph = int(pid[i])
        now = float(t)

        # 1-2. structural links declared by THIS order, resolved against prior activity
        for other in dev_cust[d]:
            if other != c:
                G.union(c, other, now)
        for other in ac_cust[ac]:
            if other != c:
                G.union(c, other, now)
        nb_active = 0
        nb_orders = 0
        for p2 in phone_nb.get(ph, ()):  # numerically adjacent numbers
            s = ph_cust.get(p2)
            if s:
                nb_active += len(s)
                nb_orders += ph_orders[p2]
                for other in s:
                    if other != c:
                        G.union(c, other, now)
        root = G.find(c)
        if not seen[c]:
            # the account's own creation date is known at checkout, so it joins
            # the component's tenure aggregate before this row is emitted
            seen[c] = True
            global_customers += 1
            G.sum_signup[root] += signup[c]
            G.min_signup[root] = min(G.min_signup[root], signup[c])

        # 3. emit
        r = out[i]
        n_prior = c_orders[c]
        r[F["value_inr"]] = val[i]
        r[F["log_value"]] = math.log1p(val[i])
        r[F["item_count"]] = itemc[i]
        r[F["promo_used"]] = promo[i]
        hh = (t // 3600) % 24
        r[F["hour_of_day"]] = hh
        r[F["day_of_week"]] = (t // 86400 + 4) % 7
        r[F["is_night"]] = 1.0 if hh < 6 else 0.0
        for ci, cname in enumerate(CATEGORIES):
            r[F["cat_" + cname]] = 1.0 if cat_i[i] == ci else 0.0

        r[F["cust_tenure_days"]] = max(now - signup[c], 0.0) / DAY
        r[F["cust_prior_orders"]] = n_prior
        r[F["cust_prior_cod"]] = c_cod[c]
        r[F["cust_cod_share"]] = (c_cod[c] + 1.0) / (n_prior + 2.0)
        r[F["cust_prior_returns"]] = c_returns[c]
        r[F["cust_return_rate"]] = (c_returns[c] + 1.0) / (n_prior + 8.0)
        r[F["cust_days_since_last"]] = (now - c_last[c]) / DAY if n_prior else 999.0
        r[F["cust_orders_d7"]] = _decay(c_d7[c], c_dts[c], now, TAU7)
        r[F["cust_orders_d30"]] = _decay(c_d30[c], c_dts[c], now, TAU30)
        mean_v = c_sum[c] / n_prior if n_prior else 0.0
        r[F["cust_mean_value"]] = mean_v
        sd = math.sqrt(c_m2[c] / (n_prior - 1)) if n_prior > 1 else 0.0
        r[F["cust_value_z"]] = (val[i] - mean_v) / sd if sd > 1e-6 else 0.0
        r[F["cust_max_value"]] = c_max[c]
        r[F["cust_distinct_devices"]] = len(c_devs[c])
        r[F["cust_distinct_addr_clusters"]] = len(c_addrs[c])

        dset = dev_cust[d]
        r[F["dev_distinct_customers"]] = len(dset) - (1 if c in dset else 0)
        r[F["dev_prior_orders"]] = dev_orders[d]
        r[F["dev_return_rate"]] = (dev_returns[d] + 1.0) / (dev_orders[d] + 8.0)
        r[F["dev_age_days"]] = (now - dev_first[d]) / DAY if d in dev_first else 0.0

        aset = ac_cust[ac]
        r[F["addr_distinct_customers"]] = len(aset) - (1 if c in aset else 0)
        r[F["addr_prior_orders"]] = ac_orders[ac]
        r[F["addr_return_rate"]] = (ac_returns[ac] + 1.0) / (ac_orders[ac] + 8.0)
        r[F["addr_variant_count"]] = len(ac_variants[ac])
        # a brand-new way of spelling an address we have already shipped to
        r[F["addr_is_new_variant_of_known_cluster"]] = \
            1.0 if (ac_variants[ac] and a not in ac_variants[ac]) else 0.0

        r[F["phone_neighbours_active"]] = nb_active
        r[F["phone_neighbour_orders"]] = nb_orders

        r[F["pin_distinct_customers"]] = len(pin_cust[pin])
        r[F["pin_prior_orders"]] = pin_orders[pin]
        r[F["pin_return_rate"]] = (pin_returns[pin] + 1.0) / (pin_orders[pin] + 8.0)

        csz = float(G.size[root])
        co = float(G.orders[root])
        r[F["comp_size"]] = csz
        r[F["comp_prior_orders"]] = co
        r[F["comp_return_rate"]] = (G.returns[root] + 1.0) / (co + 8.0)
        r[F["comp_cod_share"]] = (G.cod[root] + 1.0) / (co + 2.0)
        r[F["comp_orders_d7"]] = _decay(G.d7[root], G.d_ts[root], now, TAU7)
        r[F["comp_orders_d30"]] = _decay(G.d30[root], G.d_ts[root], now, TAU30)
        r[F["comp_growth_d7"]] = _decay(G.g7[root], G.g_ts[root], now, TAU7)
        mean_su = G.sum_signup[root] / csz if csz else signup[c]
        r[F["comp_mean_tenure_days"]] = max(now - mean_su, 0.0) / DAY
        r[F["comp_max_tenure_days"]] = max(now - G.min_signup[root], 0.0) / DAY \
            if np.isfinite(G.min_signup[root]) else 0.0
        r[F["comp_distinct_devices"]] = len(G.devs[root])
        r[F["comp_distinct_addr_clusters"]] = len(G.addrs[root])
        r[F["comp_value_per_customer"]] = G.value[root] / csz if csz else 0.0
        r[F["comp_orders_per_customer"]] = co / csz if csz else 0.0

        k = cat_i[i]
        r[F["cat_return_rate_prior"]] = (cat_returns[k] + 1.0) / (cat_orders[k] + 8.0)
        cmean = cat_value[k] / cat_orders[k] if cat_orders[k] else val[i]
        r[F["value_over_cat_mean"]] = val[i] / cmean if cmean > 1e-6 else 1.0

        tenure_d = max(r[F["cust_tenure_days"]], 1.0)
        dev_age = max(r[F["dev_age_days"]], 1.0)
        addr_age = (now - ac_first[ac]) / DAY if ac in ac_first else 0.0
        r[F["addr_age_days"]] = addr_age
        r[F["cust_orders_per_day"]] = n_prior / tenure_d
        r[F["dev_orders_per_day"]] = dev_orders[d] / dev_age
        r[F["addr_orders_per_day"]] = ac_orders[ac] / max(addr_age, 1.0)
        r[F["comp_orders_per_member_per_day"]] = co / (csz * max(
            r[F["comp_mean_tenure_days"]], 1.0)) if csz else 0.0
        r[F["pin_orders_share"]] = pin_orders[pin] / max(global_orders, 1)
        r[F["pin_cust_share"]] = len(pin_cust[pin]) / max(global_customers, 1)
        r[F["comp_orders_share"]] = co / max(global_orders, 1)

        # 4. fold this order's own behaviour into state - strictly after emission
        prev_n = c_orders[c]
        prev_mean = c_sum[c] / prev_n if prev_n else 0.0
        c_orders[c] = prev_n + 1
        c_sum[c] += val[i]
        c_m2[c] += (val[i] - prev_mean) * (val[i] - c_sum[c] / c_orders[c])
        c_max[c] = max(c_max[c], val[i])
        c_cod[c] += 1 if is_cod[i] else 0
        c_last[c] = now
        c_d7[c] = _decay(c_d7[c], c_dts[c], now, TAU7) + 1.0
        c_d30[c] = _decay(c_d30[c], c_dts[c], now, TAU30) + 1.0
        c_dts[c] = now
        c_devs[c].add(d)
        c_addrs[c].add(ac)

        if c not in dset:
            dset.add(c)
        dev_orders[d] += 1
        if d not in dev_first:
            dev_first[d] = now
        if c not in aset:
            aset.add(c)
        ac_orders[ac] += 1
        ac_variants[ac].add(a)
        if ac not in ac_first:
            ac_first[ac] = now
        global_orders += 1
        pin_cust[pin].add(c)
        pin_orders[pin] += 1
        ph_cust[ph].add(c)
        ph_orders[ph] += 1
        cat_orders[k] += 1
        cat_value[k] += val[i]

        G.orders[root] += 1
        G.cod[root] += 1 if is_cod[i] else 0
        G.value[root] += val[i]
        G.d7[root] = _decay(G.d7[root], G.d_ts[root], now, TAU7) + 1.0
        G.d30[root] = _decay(G.d30[root], G.d_ts[root], now, TAU30) + 1.0
        G.d_ts[root] = now
        G.devs[root].add(d)
        G.addrs[root].add(ac)

    df = pd.DataFrame(out, columns=FEATURE_COLUMNS)
    df.insert(0, "order_id", oid)
    return df


def load_inputs(data_dir=DATA_DIR):
    orders = pd.read_parquet(os.path.join(data_dir, "orders.parquet"))
    returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
    customers = pd.read_parquet(os.path.join(data_dir, "customers.parquet"))
    addresses = pd.read_parquet(os.path.join(data_dir, "addresses.parquet"))
    phones = pd.read_parquet(os.path.join(data_dir, "phones.parquet"))
    orders = orders[[c for c in orders.columns if c in ALLOWED_ORDER_COLS
                     or c in ("return_flag", "delivered_flag")]]
    orders = orders.drop(columns=[c for c in ("return_flag", "delivered_flag")
                                  if c in orders.columns])
    return (orders,
            customers[["customer_id", "signup_ts"]],
            addresses[["address_id", "raw_line", "pincode"]],
            phones[["phone_id", "msisdn"]],
            returns[["order_id", "return_ts"]])


def main():
    orders, customers, addresses, phones, returns = load_inputs()
    print("[feat] %d orders, %d returns" % (len(orders), len(returns)))
    feats = build_features(orders, customers, addresses, phones, returns)
    feats.to_parquet(os.path.join(DATA_DIR, "features.parquet"), index=False)
    print("[feat] wrote data/features.parquet", feats.shape)


if __name__ == "__main__":
    main()
