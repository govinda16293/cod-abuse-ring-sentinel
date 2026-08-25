"""Phase 1 - synthetic data generator.

Produces seven tables (customers, devices, addresses, phones, orders, returns,
ground_truth) plus a ring-grouped and a temporal split, then freezes and hashes
the held-out test set.

Nothing here is scraped, purchased or derived from real customer data. Names,
phone numbers and addresses are drawn from token pools and are not routable.

Framing is the merchant's defensive side throughout: we model what an abuse ring
leaves behind in an order log so that a detector can be measured against it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GEN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("SENTINEL_DATA_DIR") or os.path.join(ROOT, "data")

SEGMENTS = ["casual", "frequent", "bargain", "premium"]
SEG_P = np.array([0.45, 0.25, 0.20, 0.10])
SEG_LAMBDA = {"casual": 2.2, "frequent": 8.5, "bargain": 5.0, "premium": 6.0}
SEG_RETURN_MULT = {"casual": 1.0, "frequent": 1.15, "bargain": 1.30, "premium": 0.90}
SEG_COD_P = {"casual": 0.62, "frequent": 0.48, "bargain": 0.70, "premium": 0.30}
SEG_VALUE_MULT = {"casual": 1.0, "frequent": 1.0, "bargain": 0.75, "premium": 1.60}

BUILDING_TOKENS_A = ["Sai", "Shanti", "Green", "Lake", "Rose", "Silver", "Sun", "Ashok",
                     "Krishna", "Pearl", "Orchid", "Amber", "Neel", "Ganga", "Vasant",
                     "Palm", "Maple", "Ruby", "Indra", "Surya", "Meera", "Trident"]
BUILDING_TOKENS_B = ["Residency", "Apartments", "Heights", "Towers", "Enclave", "Nivas",
                     "Complex", "Chambers", "Villa", "Plaza", "Sadan", "Court"]
STREETS = ["MG Road", "Station Road", "Link Road", "Sector 14", "Ring Road", "Gandhi Marg",
           "Nehru Nagar", "Church Street", "Park Lane", "Market Road", "Old Post Office Road"]
CITY_TIERS = [1, 2, 3]
CITY_TIER_P = np.array([0.38, 0.37, 0.25])

RING_CATEGORY_MIX = {"apparel": 0.38, "footwear": 0.22, "electronics": 0.30,
                     "beauty": 0.10, "grocery": 0.0}


def _flat_variants(rng, flat_num, wing):
    base = str(flat_num) + wing
    opts = [base, str(flat_num) + "-" + wing, str(flat_num) + " " + wing,
            "No " + base, "No. " + base, str(flat_num) + "/" + wing, "#" + base]
    return opts[int(rng.integers(0, len(opts)))]


def _render_address(rng, flat_num, wing, bname, street, fuzz):
    """The same physical flat, written the way seven different people write it.

    The generator emits the variants; the feature layer has to re-cluster them
    without ever being shown building_id.
    """
    if not fuzz:
        return "Flat " + str(flat_num) + wing + ", " + bname + ", " + street
    flat = _flat_variants(rng, flat_num, wing)
    prefixes = ["Flat", "Flat No", "Flt", "Apt", ""]
    prefix = prefixes[int(rng.integers(0, len(prefixes)))]
    bopts = [bname, bname.replace("Apartments", "Apts"), bname.replace("Residency", "Res."),
             bname.upper(), bname.replace("Towers", "Twr")]
    bn = bopts[int(rng.integers(0, len(bopts)))]
    seps = [", ", " , ", " - ", "  "]
    sep = seps[int(rng.integers(0, len(seps)))]
    head = (prefix + " " + flat).strip()
    return head + ", " + bn + sep + street


class Generator:
    def __init__(self, cfg=GEN):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = pd.Timestamp(cfg.start_date)
        self.end = self.start + pd.DateOffset(months=cfg.months)
        self.addresses = []
        self.devices = []
        self.phones = []
        self.customers = []
        self._next = {"addr": 0, "dev": 0, "phone": 0, "cust": 0}
        self.ring_rows = []

    def _id(self, k):
        v = self._next[k]
        self._next[k] = v + 1
        return v

    def new_address(self, pincode, building_id, flat_num, wing, bname, street, fuzz=False):
        aid = self._id("addr")
        self.addresses.append(dict(
            address_id=aid,
            raw_line=_render_address(self.rng, flat_num, wing, bname, street, fuzz),
            pincode=int(pincode), building_id=int(building_id),
            flat_num=int(flat_num), wing=wing))
        return aid

    def new_device(self, first_seen):
        did = self._id("dev")
        t = ["android", "ios", "web"][int(self.rng.choice(3, p=[0.68, 0.19, 0.13]))]
        self.devices.append(dict(device_id=did, device_type=t, first_seen_ts=first_seen))
        return did

    def new_phone(self, family_id, seq):
        pid = self._id("phone")
        base = 7000000000 + (family_id * 137 + seq) % 2999999999
        self.phones.append(dict(phone_id=pid, msisdn=str(base).zfill(10),
                                phone_family_id=int(family_id)))
        return pid

    # ------------------------------------------------------------------ world
    def build_geography(self):
        rng = self.rng
        n_pin = 500
        self.pincodes = 110000 + rng.choice(np.arange(0, 90000), size=n_pin, replace=False)
        self.buildings = []
        self.pin_buildings = {}
        for pi in range(n_pin):
            ids = []
            for _ in range(int(rng.integers(25, 60))):
                bname = (BUILDING_TOKENS_A[int(rng.integers(0, len(BUILDING_TOKENS_A)))] + " " +
                         BUILDING_TOKENS_B[int(rng.integers(0, len(BUILDING_TOKENS_B)))])
                self.buildings.append(dict(pincode=int(self.pincodes[pi]), name=bname,
                                           street=STREETS[int(rng.integers(0, len(STREETS)))]))
                ids.append(len(self.buildings) - 1)
            self.pin_buildings[pi] = ids
        self.n_pin = n_pin

    def _home_address(self, fuzz=False, building_id=None, flat=None):
        rng = self.rng
        if building_id is None:
            pi = int(rng.integers(0, self.n_pin))
            blist = self.pin_buildings[pi]
            building_id = int(blist[int(rng.integers(0, len(blist)))])
        b = self.buildings[building_id]
        if flat is None:
            flat = (int(rng.integers(1, 25)), "ABCD"[int(rng.integers(0, 4))])
        aid = self.new_address(b["pincode"], building_id, flat[0], flat[1],
                               b["name"], b["street"], fuzz=fuzz)
        return aid, building_id, flat

    def fuzz_variant_of(self, address_id):
        """Another spelling of an address we already have.

        Used for abuse rings AND for legitimate households. Real families type
        their own address differently every time - if only rings produced
        variants, "this cluster has many spellings" would be a giveaway that
        exists nowhere outside this generator, and R2 recall would be fake.
        """
        a = self.addresses[int(address_id)]
        b = self.buildings[a["building_id"]]
        return self.new_address(a["pincode"], a["building_id"], a["flat_num"], a["wing"],
                                b["name"], b["street"], fuzz=True)

    def new_customer(self, signup_ts, segment=None, address_id=None, device_id=None,
                     phone_id=None, household_id=-1, **extra):
        cid = self._id("cust")
        seg = segment or SEGMENTS[int(self.rng.choice(4, p=SEG_P))]
        if address_id is None:
            address_id = self._home_address()[0]
        if device_id is None:
            device_id = self.new_device(signup_ts)
        if phone_id is None:
            phone_id = self.new_phone(family_id=cid, seq=0)
        row = dict(customer_id=cid, signup_ts=signup_ts, segment=seg,
                   primary_address_id=address_id, primary_device_id=device_id,
                   primary_phone_id=phone_id, household_id=household_id,
                   is_hard_neg=0, hn_type="none", return_mult=1.0,
                   cod_p_override=np.nan, cat_override="none", lam_mult=1.0,
                   is_ring_member=0, ring_id=-1)
        row.update(extra)
        self.customers.append(row)
        return cid

    def _rand_signup(self, lo=None, hi=None):
        lo = lo if lo is not None else self.start - pd.Timedelta(days=760)
        hi = hi if hi is not None else self.end - pd.Timedelta(days=5)
        secs = max(int((hi - lo).total_seconds()), 86400)
        return lo + pd.Timedelta(seconds=int(self.rng.integers(0, secs)))

    # ------------------------------------------------------------- population
    def build_population(self):
        """Benign customers plus hard-negative households.

        Hard negatives exist so that precision means something. Each type is
        built to trip one specific naive rule:
          H1 -> "same address AND same device"      H2 -> "pincode concentration"
          H3 -> "return rate above 40%"             H4 -> "device seen on 2 accounts"
        """
        cfg, rng = self.cfg, self.rng
        n_hn = int(cfg.n_customers * cfg.hard_neg_frac)
        counts = {k: int(n_hn * v) for k, v in cfg.hard_neg_mix.items()}
        hh = 0

        made = 0
        while made < counts["H1_joint_family"]:
            k = int(rng.integers(3, 7))
            aid = self._home_address()[0]
            shared_dev = self.new_device(self._rand_signup())
            for _ in range(k):
                dev = shared_dev if rng.random() < 0.62 else self.new_device(self._rand_signup())
                # same flat, spelled differently by different family members
                a_j = self.fuzz_variant_of(aid) if rng.random() < 0.60 else aid
                self.new_customer(self._rand_signup(), address_id=a_j, device_id=dev,
                                  household_id=hh, is_hard_neg=1, hn_type="H1_joint_family")
            hh += 1
            made += k

        made = 0
        while made < counts["H2_hostel_pincode"]:
            k = int(rng.integers(15, 41))
            bid = self._home_address()[1]
            for j in range(k):
                aid = self._home_address(building_id=bid, flat=(j + 1, "A"))[0]
                self.new_customer(self._rand_signup(), address_id=aid, household_id=hh,
                                  is_hard_neg=1, hn_type="H2_hostel_pincode")
            hh += 1
            made += k

        for _ in range(counts["H3_high_returner"]):
            su = self._rand_signup(hi=self.start - pd.Timedelta(days=500))
            self.new_customer(su, segment="frequent", household_id=hh, is_hard_neg=1,
                              hn_type="H3_high_returner",
                              return_mult=float(rng.uniform(3.0, 4.4)),
                              cod_p_override=float(rng.uniform(0.22, 0.34)),
                              cat_override="fashion", lam_mult=1.9)
            hh += 1

        made = 0
        while made < counts["H4_device_resale"]:
            dev = self.new_device(self._rand_signup())
            su1 = self._rand_signup(hi=self.end - pd.Timedelta(days=220))
            self.new_customer(su1, device_id=dev, household_id=hh, is_hard_neg=1,
                              hn_type="H4_device_resale")
            su2 = su1 + pd.Timedelta(days=int(rng.integers(95, 190)))
            self.new_customer(su2, device_id=dev, household_id=hh, is_hard_neg=1,
                              hn_type="H4_device_resale")
            hh += 1
            made += 2

        for _ in range(cfg.n_customers - n_hn):
            self.new_customer(self._rand_signup())
        self.n_households = hh

    # ----------------------------------------------------- benign order stream
    def generate_benign_orders(self):
        cfg, rng = self.cfg, self.rng
        cust = pd.DataFrame(self.customers)
        n = len(cust)

        has_sec_dev = rng.random(n) < 0.25
        has_sec_addr = rng.random(n) < 0.15
        sec_dev = np.full(n, -1, dtype=np.int64)
        sec_addr = np.full(n, -1, dtype=np.int64)
        for i in np.flatnonzero(has_sec_dev):
            sec_dev[i] = self.new_device(cust.signup_ts.iloc[int(i)])
        for i in np.flatnonzero(has_sec_addr):
            # half of second addresses are a re-typing of the customer's own
            # address (typo / autofill drift), half are a genuinely different
            # place (office, parents). Without the first half, "a new spelling
            # of a known address" would be an abuse-only signal.
            sec_addr[i] = (self.fuzz_variant_of(cust.primary_address_id.iloc[int(i)])
                           if rng.random() < 0.5 else self._home_address()[0])

        lam = cust.segment.map(SEG_LAMBDA).to_numpy() * cust.lam_mult.to_numpy()
        if cfg.flat_signup:
            # DIAGNOSTIC MODE. In the default generator every customer draws the
            # same expected order count regardless of when they signed up, so a
            # customer who joins five days before the window ends crams all of
            # them into those five days. That inflates both order volume and the
            # new-account share at late calendar times, which is exactly the
            # confound suspected of manufacturing the temporal precision drop.
            #
            # Here each customer's rate is scaled by the fraction of the window
            # they were actually around for, so orders per calendar day are flat.
            su = cust.signup_ts.to_numpy().astype("datetime64[s]").astype(np.int64)
            t0 = np.int64(self.start.value // 10**9)
            t1 = np.int64(self.end.value // 10**9)
            exposure = np.clip((t1 - np.maximum(su, t0)) / float(t1 - t0), 0.0, 1.0)
            lam = lam * exposure
        scale = cfg.n_orders_target / max(lam.sum(), 1.0)
        counts = rng.poisson(lam * scale)
        cidx = np.repeat(np.arange(n), counts)
        m = len(cidx)

        signup = cust.signup_ts.to_numpy()[cidx].astype("datetime64[s]").astype(np.int64)
        lo = np.maximum(signup, np.int64(self.start.value // 10**9))
        hi = np.int64(self.end.value // 10**9)
        span = np.maximum(hi - lo, 3600)
        ots = lo + (rng.random(m) * span).astype(np.int64)

        cats = list(cfg.category_mix)
        cat_p = np.array([cfg.category_mix[c] for c in cats])
        cat_i = rng.choice(len(cats), size=m, p=cat_p / cat_p.sum())
        fashion = cust.cat_override.to_numpy()[cidx] == "fashion"
        cat_i = np.where(fashion,
                         rng.choice([cats.index("apparel"), cats.index("footwear")],
                                    size=m, p=[0.68, 0.32]), cat_i)

        med = np.array([cfg.category_value[c][0] for c in cats])[cat_i]
        sig = np.array([cfg.category_value[c][1] for c in cats])[cat_i]
        vmult = cust.segment.map(SEG_VALUE_MULT).to_numpy()[cidx]
        value = np.clip(np.round(med * np.exp(sig * rng.standard_normal(m)) * vmult), 99, 180000)

        cod_base = cust.segment.map(SEG_COD_P).to_numpy()
        ov = cust.cod_p_override.to_numpy(dtype=float)
        cod_base = np.where(np.isnan(ov), cod_base, ov)[cidx]
        cod_adj = 1.0 - 0.30 / (1.0 + np.exp(-(value - 9000.0) / 3000.0))
        is_cod = rng.random(m) < np.clip(cod_base * cod_adj, 0.01, 0.98)

        ret_base = np.array([cfg.category_return_rate[c] for c in cats])[cat_i]
        ret_p = (ret_base * cust.segment.map(SEG_RETURN_MULT).to_numpy()[cidx]
                 * cust.return_mult.to_numpy()[cidx] * np.where(is_cod, 1.15, 1.0))
        returned = rng.random(m) < np.clip(ret_p, 0.0, 0.93)
        ret_delay = (rng.integers(3, 26, size=m) * 86400).astype(np.int64)

        use_d = (rng.random(m) < 0.12) & (sec_dev[cidx] >= 0)
        use_a = (rng.random(m) < 0.10) & (sec_addr[cidx] >= 0)
        dev = np.where(use_d, sec_dev[cidx], cust.primary_device_id.to_numpy()[cidx])
        addr = np.where(use_a, sec_addr[cidx], cust.primary_address_id.to_numpy()[cidx])

        reasons = np.array(["size", "damaged", "not_as_described", "wrong_item"])
        df = pd.DataFrame(dict(
            customer_id=cust.customer_id.to_numpy()[cidx].astype(np.int64),
            device_id=dev.astype(np.int64), address_id=addr.astype(np.int64),
            phone_id=cust.primary_phone_id.to_numpy()[cidx].astype(np.int64),
            order_ts=pd.to_datetime(ots, unit="s"),
            payment_mode=np.where(is_cod, "COD", "prepaid"),
            category=np.array(cats)[cat_i],
            value_inr=value.astype(np.float64),
            item_count=(1 + rng.poisson(0.6, size=m)).astype(np.int16),
            promo_used=(rng.random(m) < 0.30).astype(np.int8),
            delivered_flag=(rng.random(m) < 0.96).astype(np.int8),
            return_flag=returned.astype(np.int8),
            return_ts=pd.to_datetime(np.where(returned, ots + ret_delay, np.nan), unit="s"),
            return_reason=np.where(returned, reasons[rng.choice(4, size=m,
                                   p=[0.42, 0.21, 0.27, 0.10])], None),
            refund_mode=np.where(returned & is_cod, "COD_cash_back",
                                 np.where(returned, "wallet", None)),
            outcome=np.where(returned, np.array(["refunded", "rejected"])[
                rng.choice(2, size=m, p=[0.91, 0.09])], None),
            ring_id=np.full(m, -1, dtype=np.int64),
            ring_type=np.full(m, "none"),
            is_abuse_true=np.zeros(m, dtype=np.int8),
        ))
        self.benign_orders = df
        return df

    # ------------------------------------------------------------------ rings
    def _campaign_order(self, cid, dev, addr, phone, ts, ring_id, ring_type, is_abuse,
                        value_mult=1.0, force_cod=None):
        rng = self.rng
        cats = [c for c, p in RING_CATEGORY_MIX.items() if p > 0]
        p = np.array([RING_CATEGORY_MIX[c] for c in cats])
        cat = cats[int(rng.choice(len(cats), p=p / p.sum()))]
        med, sig = self.cfg.category_value[cat]
        value = float(np.clip(round(med * np.exp(sig * rng.standard_normal()) * value_mult),
                              99, 180000))
        cod = force_cod if force_cod is not None else (rng.random() < self.cfg.cod_share_ring)
        ret = rng.random() < (self.cfg.ring_return_rate if is_abuse else 0.12)
        rts = ts + pd.Timedelta(days=int(rng.integers(2, 13))) if ret else pd.NaT
        return dict(
            customer_id=cid, device_id=dev, address_id=addr, phone_id=phone, order_ts=ts,
            payment_mode="COD" if cod else "prepaid", category=cat, value_inr=value,
            item_count=int(1 + rng.poisson(0.4)), promo_used=int(rng.random() < 0.44),
            delivered_flag=int(rng.random() < 0.97), return_flag=int(ret), return_ts=rts,
            return_reason=(["not_as_described", "damaged", "size", "wrong_item"]
                           [int(rng.choice(4, p=[0.46, 0.24, 0.20, 0.10]))] if ret else None),
            refund_mode=("COD_cash_back" if (ret and cod) else ("wallet" if ret else None)),
            outcome=(["refunded", "rejected"][int(rng.choice(2, p=[0.93, 0.07]))] if ret else None),
            ring_id=ring_id, ring_type=ring_type, is_abuse_true=int(is_abuse))

    def plant_rings(self, n_campaign_orders_target):
        """Plant abuse rings with varied sharing signatures.

        Each ring is a group of accounts that co-ordinate returnable COD orders.
        The five signatures differ in WHICH identifier they share, which is the
        whole point: a detector that only knows one of them will look great on a
        blended recall number and fall over on the others.
        """
        cfg, rng = self.cfg, self.rng
        types = list(cfg.ring_type_mix)
        tp = np.array([cfg.ring_type_mix[t] for t in types])
        tp = tp / tp.sum()
        rows = []
        ring_id = 0
        placed = 0
        while placed < n_campaign_orders_target:
            rtype = types[int(rng.choice(len(types), p=tp))]
            k = int(np.clip(round(float(rng.lognormal(cfg.ring_size_lognorm_mu,
                                                      cfg.ring_size_lognorm_sigma))),
                            cfg.ring_size_min, cfg.ring_size_max))
            camp_start = self.start + pd.Timedelta(
                seconds=int(rng.integers(0, int((self.end - self.start).total_seconds() * 0.94))))
            if rtype == "R4_burst_value":
                window_d = int(rng.integers(3, 11))
                tenure_lo, tenure_hi = 1, 14
            else:
                window_d = int(rng.integers(14, 57))
                tenure_lo, tenure_hi = 5, 90

            shared_dev = [self.new_device(camp_start) for _ in
                          range(int(rng.integers(1, 3)))] if rtype == "R1_shared_device" else None
            if rtype in ("R2_address_fuzz",):
                base_aid, base_bid, base_flat = self._home_address()
            if rtype == "R3_phone_family":
                fam = 900000 + ring_id
                fam_pin_building = self._home_address()[1]
            if rtype == "R5_hybrid":
                mech = list(rng.choice(["device", "address", "phone"], size=2, replace=False))
                hyb_dev = self.new_device(camp_start) if "device" in mech else None
                if "address" in mech:
                    h_aid, h_bid, h_flat = self._home_address()
                hyb_fam = 950000 + ring_id if "phone" in mech else None

            members = []
            for j in range(k):
                signup = camp_start - pd.Timedelta(days=int(rng.integers(tenure_lo, tenure_hi)),
                                                   seconds=int(rng.integers(0, 86400)))
                dev = phone = None
                addr = None
                if rtype == "R1_shared_device":
                    dev = int(shared_dev[j % len(shared_dev)])
                elif rtype == "R2_address_fuzz":
                    if rng.random() < cfg.address_fuzz_rate:
                        b = self.buildings[base_bid]
                        addr = self.new_address(b["pincode"], base_bid, base_flat[0],
                                                base_flat[1], b["name"], b["street"], fuzz=True)
                    else:
                        addr = base_aid
                elif rtype == "R3_phone_family":
                    phone = self.new_phone(family_id=fam, seq=j)
                    if rng.random() < 0.5:
                        addr = self._home_address(building_id=fam_pin_building)[0]
                elif rtype == "R5_hybrid":
                    if hyb_dev is not None and rng.random() < 0.60:
                        dev = int(hyb_dev)
                    if "address" in mech and rng.random() < 0.50:
                        b = self.buildings[h_bid]
                        addr = self.new_address(b["pincode"], h_bid, h_flat[0], h_flat[1],
                                                b["name"], b["street"], fuzz=True)
                    if hyb_fam is not None and rng.random() < 0.55:
                        phone = self.new_phone(family_id=hyb_fam, seq=j)
                # R4_burst_value deliberately shares NO identifier. It is the
                # purely behavioural signature (new accounts, tight window, high
                # value, COD) and is meant to be the easy case, so that the
                # per-ring-type recall table shows an honest spread.
                cid = self.new_customer(signup, segment="casual", address_id=addr,
                                        device_id=dev, phone_id=phone, household_id=-1,
                                        is_ring_member=1, ring_id=ring_id)
                c = self.customers[-1]
                members.append((cid, c["primary_device_id"], c["primary_address_id"],
                                c["primary_phone_id"], signup))

            vmult = 2.1 if rtype == "R4_burst_value" else 1.45
            for (cid, dev, addr, phone, signup) in members:
                # a small number of ordinary orders from the same accounts, label 0.
                # They exist so the model cannot treat "account belongs to a ring"
                # as equivalent to "this order is abuse".
                n_filler = int(rng.integers(0, 3)) if rtype == "R5_hybrid" else \
                    (1 if rng.random() < 0.30 else 0)
                for _ in range(n_filler):
                    ts = signup + pd.Timedelta(seconds=int(rng.integers(3600, 86400 * 12)))
                    if ts >= self.end:
                        continue
                    rows.append(self._campaign_order(cid, dev, addr, phone, ts, ring_id, rtype,
                                                     is_abuse=0, value_mult=0.45,
                                                     force_cod=False))
                n_ord = int(rng.integers(2, 5)) if rtype != "R4_burst_value" else \
                    int(rng.integers(1, 4))
                for _ in range(n_ord):
                    ts = camp_start + pd.Timedelta(
                        seconds=int(rng.integers(0, max(window_d * 86400, 3600))))
                    if ts >= self.end:
                        ts = self.end - pd.Timedelta(hours=int(rng.integers(1, 72)))
                    rows.append(self._campaign_order(cid, dev, addr, phone, ts, ring_id, rtype,
                                                     is_abuse=1, value_mult=vmult))
                    placed += 1
            ring_id += 1
        self.n_rings = ring_id
        self.ring_orders = pd.DataFrame(rows)
        return self.ring_orders


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def group_key(row):
    """Split unit. A ring is atomic; a household is atomic; otherwise a customer.

    Row-level splitting would put ring members on both sides and inflate recall.
    Households are atomic for the same reason in reverse: if half a joint family
    is in train, the model can memorise the address instead of learning the
    behaviour, and the hard negative stops being hard.
    """
    if row["ring_id"] >= 0:
        return "ring_" + str(int(row["ring_id"]))
    if row["household_id"] >= 0:
        return "hh_" + str(int(row["household_id"]))
    return "cust_" + str(int(row["customer_id"]))


def make_splits(orders, cfg=GEN, seed=None):
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 1)
    keys = orders["group_key"].unique()
    perm = rng.permutation(len(keys))
    keys = keys[perm]
    n = len(keys)
    n_tr = int(n * cfg.split_train)
    n_va = int(n * cfg.split_val)
    assign = {}
    for i, k in enumerate(keys):
        assign[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return orders["group_key"].map(assign)


def make_temporal_split(orders, start, months_train=12):
    """Second evaluation: train on months 1-12, test on months 13-18.

    Rings must still not cross the boundary, so any RING with campaign orders in
    the train period is dropped from the temporal test set - otherwise the model
    has already seen that ring's members labelled as abuse.

    Ordinary customers and hard-negative households are deliberately NOT dropped
    when they span the boundary. That is the real production setting: on 1 Jan
    you already know your existing customers. Dropping them would leave a
    temporal test set made only of accounts whose first ever order falls after
    the boundary, which is a new-account-heavy population and not a fair
    comparison against the ring-grouped split.
    """
    boundary = start + pd.DateOffset(months=months_train)
    is_ring = orders["ring_id"].to_numpy() >= 0
    rings_before = set(orders.loc[is_ring & (orders.order_ts < boundary),
                                  "ring_id"].unique().tolist())
    side = np.where(orders.order_ts < boundary, "train", "test")
    straddle = is_ring & orders["ring_id"].isin(rings_before).to_numpy() & (side == "test")
    side = np.where(straddle, "drop", side)
    return (pd.Series(side, index=orders.index), boundary,
            int(orders.loc[straddle, "ring_id"].nunique()),
            int(straddle.sum()))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = GEN
    if os.environ.get("SENTINEL_FLAT_SIGNUP") == "1":
        cfg.flat_signup = True
        print("[gen] FLAT-SIGNUP DIAGNOSTIC MODE - not the headline dataset")
    if os.environ.get("SENTINEL_QUICK") == "1":
        # smoke-test size. Ring counts get small, so per-ring-type recall from a
        # quick run is noisy and must not be quoted.
        cfg.n_customers = 12_000
        cfg.n_orders_target = 50_000
    g = Generator(cfg)
    print("[gen] building geography ...")
    g.build_geography()
    print("[gen] building population ...")
    g.build_population()
    print("[gen] generating benign order stream ...")
    benign = g.generate_benign_orders()

    benign_cod = int((benign.payment_mode == "COD").sum())
    t = cfg.abuse_rate_of_cod_orders
    abuse_cod_needed = t / (1.0 - t) * benign_cod
    campaign_needed = int(round(abuse_cod_needed / cfg.cod_share_ring))
    print("[gen] benign COD orders=%d -> planting ~%d campaign orders" %
          (benign_cod, campaign_needed))
    ring = g.plant_rings(campaign_needed)

    orders = pd.concat([benign, ring], ignore_index=True)
    orders = orders.sort_values("order_ts", kind="mergesort").reset_index(drop=True)
    orders["order_id"] = np.arange(len(orders), dtype=np.int64)

    customers = pd.DataFrame(g.customers)
    addresses = pd.DataFrame(g.addresses)
    devices = pd.DataFrame(g.devices)
    phones = pd.DataFrame(g.phones)

    orders = orders.merge(customers[["customer_id", "household_id", "signup_ts", "segment",
                                     "is_hard_neg", "hn_type"]],
                          on="customer_id", how="left")
    orders["group_key"] = [group_key(r) for r in
                           orders[["ring_id", "household_id", "customer_id"]].to_dict("records")]

    # ---- label noise -------------------------------------------------------
    # Applied only to COD orders: those are the ones a returns-abuse ops team
    # actually adjudicates, so they are the only ones whose labels can be wrong.
    rng = np.random.default_rng(cfg.seed + 7)
    obs = orders["is_abuse_true"].to_numpy().copy()
    cod_mask = (orders.payment_mode == "COD").to_numpy()
    pos = np.flatnonzero((obs == 1) & cod_mask)
    flip_fn = pos[rng.random(len(pos)) < cfg.label_noise_fn]
    obs[flip_fn] = 0

    n_fp = int(round(len(pos) * cfg.label_noise_fp_rel))
    hn_mask = (orders.is_hard_neg.to_numpy() == 1) & (obs == 0) & cod_mask
    other_mask = (orders.is_hard_neg.to_numpy() == 0) & (obs == 0) & cod_mask
    n_hn = min(int(round(n_fp * cfg.label_noise_fp_from_hardneg)), int(hn_mask.sum()))
    n_ot = min(n_fp - n_hn, int(other_mask.sum()))
    flip_fp = np.concatenate([
        rng.choice(np.flatnonzero(hn_mask), size=n_hn, replace=False),
        rng.choice(np.flatnonzero(other_mask), size=n_ot, replace=False)])
    obs[flip_fp] = 1
    orders["is_abuse"] = obs.astype(np.int8)

    split = make_splits(orders, cfg)
    orders["split"] = split.to_numpy()
    tsplit, boundary, n_straddle_rings, n_straddle_orders = make_temporal_split(orders, g.start)
    orders["temporal_split"] = tsplit.to_numpy()

    truth_cols = ["order_id", "customer_id", "ring_id", "ring_type", "is_abuse_true",
                  "is_abuse", "household_id", "hn_type", "group_key", "split",
                  "temporal_split"]
    ground_truth = orders[truth_cols].copy()

    returns = orders.loc[orders.return_flag == 1,
                         ["order_id", "return_ts", "return_reason", "refund_mode",
                          "outcome"]].copy()

    # ground truth (ring_id / ring_type / is_abuse*) lives ONLY in ground_truth.
    order_cols = ["order_id", "customer_id", "device_id", "address_id", "phone_id", "order_ts",
                  "payment_mode", "category", "value_inr", "item_count", "promo_used",
                  "delivered_flag", "return_flag"]
    orders_public = orders[order_cols].copy()

    addresses.to_parquet(os.path.join(DATA_DIR, "addresses.parquet"), index=False)
    devices.to_parquet(os.path.join(DATA_DIR, "devices.parquet"), index=False)
    phones.to_parquet(os.path.join(DATA_DIR, "phones.parquet"), index=False)
    customers.drop(columns=["cod_p_override"]).to_parquet(
        os.path.join(DATA_DIR, "customers.parquet"), index=False)
    orders_public.to_parquet(os.path.join(DATA_DIR, "orders.parquet"), index=False)
    returns.to_parquet(os.path.join(DATA_DIR, "returns.parquet"), index=False)
    ground_truth.to_parquet(os.path.join(DATA_DIR, "ground_truth.parquet"), index=False)

    cod = orders.payment_mode == "COD"
    cust_abuse = ground_truth.groupby("customer_id").is_abuse_true.max()
    cod_cust = set(orders.loc[cod, "customer_id"].unique().tolist())
    stats = dict(
        n_orders=int(len(orders)), n_customers=int(len(customers)),
        n_addresses=int(len(addresses)), n_devices=int(len(devices)),
        n_phones=int(len(phones)), n_rings=int(g.n_rings),
        n_households=int(g.n_households),
        window=[str(g.start.date()), str(g.end.date())],
        # ---- scoreable population = COD orders. The detector never sees prepaid.
        cod_orders=int(cod.sum()),
        cod_share_realised=float(cod.mean()),
        abuse_orders_true=int(orders.is_abuse_true.sum()),
        abuse_orders_observed=int(orders.is_abuse.sum()),
        abuse_rate_all_orders_observed=float(orders.is_abuse.mean()),
        abuse_rate_cod_orders_observed=float(orders.loc[cod, "is_abuse"].mean()),
        abuse_rate_cod_orders_true=float(orders.loc[cod, "is_abuse_true"].mean()),
        abuse_rate_prepaid_orders_true=float(orders.loc[~cod, "is_abuse_true"].mean()),
        abuse_orders_on_prepaid_true=int(orders.loc[~cod, "is_abuse_true"].sum()),
        abuse_rate_customers_all=float((cust_abuse > 0).mean()),
        abuse_rate_customers_cod_active=float(
            (cust_abuse.loc[cust_abuse.index.isin(cod_cust)] > 0).mean()),
        hard_negative_customers=int((customers.is_hard_neg == 1).sum()),
        hard_negative_share=float((customers.is_hard_neg == 1).mean()),
        label_noise_fn=cfg.label_noise_fn,
        label_noise_fp_rel=cfg.label_noise_fp_rel,
        labels_flipped_to_0=int(len(flip_fn)), labels_flipped_to_1=int(len(flip_fp)),
        label_noise_share_of_positive_labels=float(
            len(flip_fp) / max(int(orders.is_abuse.sum()), 1)),
        temporal_boundary=str(boundary.date()),
        temporal_rings_dropped=int(n_straddle_rings),
        temporal_orders_dropped=int(n_straddle_orders),
        split_counts={k: int(v) for k, v in orders.split.value_counts().items()},
        temporal_counts={k: int(v) for k, v in orders.temporal_split.value_counts().items()},
        ring_type_orders={k: int(v) for k, v in
                          orders.loc[orders.is_abuse_true == 1, "ring_type"]
                          .value_counts().items()},
        seed=cfg.seed,
    )
    with open(os.path.join(DATA_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- freeze + hash the test set ---------------------------------------
    test_ids = np.sort(ground_truth.loc[ground_truth.split == "test", "order_id"].to_numpy())
    tpath = os.path.join(DATA_DIR, "test_order_ids.npy")
    np.save(tpath, test_ids)
    h = hashlib.sha256()
    with open(tpath, "rb") as f:
        h.update(f.read())
    payload = orders_public.loc[orders_public.order_id.isin(test_ids)].sort_values("order_id")
    h2 = hashlib.sha256(pd.util.hash_pandas_object(payload, index=False).values.tobytes())
    with open(os.path.join(DATA_DIR, "test_set.sha256"), "w") as f:
        f.write("test_order_ids.npy  sha256=" + h.hexdigest() + "\n")
        f.write("test_rows_content   sha256=" + h2.hexdigest() + "\n")
        f.write("n_test_orders=" + str(len(test_ids)) + "\n")
    print(json.dumps(stats, indent=2))
    print("[gen] frozen test set hashed -> data/test_set.sha256")


if __name__ == "__main__":
    main()
