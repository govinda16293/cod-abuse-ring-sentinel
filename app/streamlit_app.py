"""Phase 6 - thin review UI.

Three things a risk analyst actually needs: the batch, the queue, and the reason.
Deliberately unstyled - the brief says do not spend time on polish, and every
minute here is a minute not spent on the metrics that get questioned.

Everything shown is read from artifacts/ produced by the pipeline. The UI never
scores, never thresholds and never re-fits; it only renders decisions already
made, so nothing you can click here can change a number in the README.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts")
FIG = os.path.join(ART, "figures")

st.set_page_config(page_title="COD abuse-ring sentinel", layout="wide")

BAND_COLOR = {"auto_action": "#c0392b", "review": "#d68910", "pass": "#5d6d7e"}


@st.cache_data
def load():
    scored = pd.read_parquet(os.path.join(ART, "scored_test.parquet"))
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(ART, "explanation_payloads.json")) as f:
        payloads = {int(p["order"]["order_id"]): p for p in json.load(f)}
    ring = pd.read_csv(os.path.join(ART, "recall_by_ring_type.csv"))
    fps = pd.read_csv(os.path.join(ART, "false_positives_by_population.csv"))
    return scored, metrics, payloads, ring, fps


def missing_artifacts():
    need = ["scored_test.parquet", "metrics.json", "explanation_payloads.json",
            "recall_by_ring_type.csv", "false_positives_by_population.csv"]
    return [n for n in need if not os.path.exists(os.path.join(ART, n))]


miss = missing_artifacts()
if miss:
    st.error("Missing artifacts: %s\n\nRun `python run_all.py` first." % ", ".join(miss))
    st.stop()

scored, metrics, payloads, ring, fps = load()
rg = metrics["ring_grouped"]
tm = metrics["temporal"]
pol = metrics["policy"]["ring_grouped"]

st.title("COD abuse-ring sentinel")
st.caption("Defence-only detector for cash-on-delivery return abuse. All figures below "
           "are from the frozen held-out test set, scored once, with thresholds selected "
           "on validation.")

c = st.columns(6)
c[0].metric("PR-AUC", "%.3f" % rg["pr_auc"])
c[1].metric("Precision (auto-action)", "%.3f" % rg["precision_auto_action"])
c[2].metric("Recall (auto-action)", "%.3f" % rg["recall_auto_action"])
c[3].metric("Review queue", "%.2f%%" % (100 * rg["review_share"]))
c[4].metric("Cost / 1000 orders", "Rs %.0f" % rg["cost_per_1000_policy"])
c[5].metric("vs. no detector", "-%.1f%%" % rg["saving_vs_do_nothing_pct"])

tab_batch, tab_queue, tab_order, tab_metrics = st.tabs(
    ["Batch", "Review queue", "Order detail", "Metrics"])

# ---------------------------------------------------------------- batch view
with tab_batch:
    st.subheader("Scored batch - held-out test set")
    left, right = st.columns([1, 3])
    with left:
        bands = st.multiselect("Decision band", ["auto_action", "review", "pass"],
                               default=["auto_action", "review"])
        min_p = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
        cats = st.multiselect("Category", sorted(scored.category.unique()),
                              default=list(sorted(scored.category.unique())))
        vmax = float(scored.value_inr.max())
        vrange = st.slider("Order value (INR)", 0.0, vmax, (0.0, vmax))
        st.markdown("**Bands** (selected on validation)")
        st.code("pass    p < %.4f\nreview  %.4f <= p < %.4f\naction  p >= %.4f"
                % (pol["band_lo"], pol["band_lo"], pol["band_hi"], pol["band_hi"]))

    view = scored[scored.band.isin(bands) & (scored.p >= min_p)
                  & scored.category.isin(cats)
                  & scored.value_inr.between(*vrange)]
    view = view.sort_values("p", ascending=False)
    with right:
        st.write("%d of %d orders" % (len(view), len(scored)))
        st.dataframe(
            view[["order_id", "order_ts", "value_inr", "category", "p", "band",
                  "is_abuse"]].head(600).rename(columns={"is_abuse": "label(held-out)"}),
            use_container_width=True, height=560)
    st.caption("`label(held-out)` is shown only because this is an offline evaluation "
               "harness. In production the reviewer would not see it.")

# ---------------------------------------------------------- review queue view
with tab_queue:
    st.subheader("Human review queue")
    q = scored[scored.band == "review"].sort_values("p", ascending=False)
    st.write("**%d orders** (%.2f%% of batch). Capacity constraint used when selecting "
             "bands: %.0f%%." % (len(q), 100 * len(q) / len(scored),
                                 100 * metrics["policy"]["ring_grouped"]
                                 ["review_capacity_constraint"]))
    st.write("Of these, %d are abusive in the held-out labels - a %.1f%% hit rate for "
             "the reviewer." % (int(q.is_abuse.sum()),
                                100 * q.is_abuse.mean() if len(q) else 0.0))
    st.dataframe(q[["order_id", "order_ts", "value_inr", "category", "p", "is_abuse"]],
                 use_container_width=True, height=520)

# ---------------------------------------------------------- order detail view
with tab_order:
    st.subheader("Per-order explanation")
    flagged = scored[scored.band != "pass"].sort_values("p", ascending=False)
    oid = st.selectbox("Flagged order", flagged.order_id.tolist(),
                       format_func=lambda x: "order %d" % x)
    row = scored[scored.order_id == oid].iloc[0]
    pl = payloads.get(int(oid))

    a, b = st.columns([1, 2])
    with a:
        st.markdown("### Decision")
        st.markdown("<span style='color:%s;font-size:1.4rem;font-weight:600'>%s</span>"
                    % (BAND_COLOR[row.band], row.band.replace("_", " ").upper()),
                    unsafe_allow_html=True)
        st.metric("Calibrated P(abuse)", "%.4f" % row.p)
        st.write("**Order** %d" % row.order_id)
        st.write("**Placed** %s" % row.order_ts)
        st.write("**Value** Rs %.0f" % row.value_inr)
        st.write("**Category** %s" % row.category)
        st.write("**Payment** COD")
        st.write("**Held-out label** %s" % ("abuse" if row.is_abuse else "legit"))
        if row.ring_type not in (None, "none"):
            st.write("**Ring signature** %s" % row.ring_type)

    with b:
        if pl is None:
            st.warning("No attribution payload for this order.")
        else:
            st.markdown("### Evidence (TreeSHAP, top contributors)")
            ev = pd.DataFrame(pl["evidence"])
            if len(ev):
                st.dataframe(ev[["sentence", "feature", "value", "contribution"]],
                             use_container_width=True, hide_index=True)
                st.bar_chart(ev.set_index("feature")["contribution"])
            if pl["counter_evidence"]:
                st.markdown("### Evidence pointing the other way")
                st.dataframe(pd.DataFrame(pl["counter_evidence"])
                             [["sentence", "feature", "contribution"]],
                             use_container_width=True, hide_index=True)
            st.markdown("### What this does not establish")
            for cv in pl["caveats"]:
                st.markdown("- " + cv)

            st.markdown("### Reviewer justification")
            use_llm = st.checkbox(
                "Use the LLM writer (needs ANTHROPIC_API_KEY)",
                value=bool(os.environ.get("ANTHROPIC_API_KEY")))
            if st.button("Write justification"):
                from explain import write_explanation
                with st.spinner("writing..."):
                    text, src = write_explanation(pl, use_llm=use_llm)
                st.info(text)
                st.caption("writer: %s" % src)
            st.caption("The language model never scores the order. It is handed the "
                       "payload above and asked to render it as prose; its output has no "
                       "path back into the decision.")

# --------------------------------------------------------------- metrics view
with tab_metrics:
    st.subheader("Held-out results")
    st.markdown("#### Ring-grouped split (primary) vs temporal split (months 13-18)")
    comp = pd.DataFrame([
        {"metric": "orders scored", "ring-grouped": rg["n_orders"], "temporal": tm["n_orders"]},
        {"metric": "abuse orders", "ring-grouped": rg["n_abuse"], "temporal": tm["n_abuse"]},
        {"metric": "base rate", "ring-grouped": round(rg["base_rate"], 4),
         "temporal": round(tm["base_rate"], 4)},
        {"metric": "PR-AUC", "ring-grouped": round(rg["pr_auc"], 4),
         "temporal": round(tm["pr_auc"], 4)},
        {"metric": "precision (auto-action)", "ring-grouped": round(rg["precision_auto_action"], 4),
         "temporal": round(tm["precision_auto_action"], 4)},
        {"metric": "recall (auto-action)", "ring-grouped": round(rg["recall_auto_action"], 4),
         "temporal": round(tm["recall_auto_action"], 4)},
        {"metric": "cost per 1000 (INR)", "ring-grouped": round(rg["cost_per_1000_policy"]),
         "temporal": round(tm["cost_per_1000_policy"])},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("#### Recall by ring signature")
    st.dataframe(ring, use_container_width=True, hide_index=True)
    st.markdown("#### False positives by legitimate population")
    st.dataframe(fps, use_container_width=True, hide_index=True)

    cols = st.columns(3)
    for col, name, cap in [
            (cols[0], "pr_curve_ring_grouped.png", "Precision-recall"),
            (cols[1], "reliability_ring_grouped.png", "Reliability"),
            (cols[2], "cost_curve_ring_grouped.png", "Cost curve")]:
        path = os.path.join(FIG, name)
        if os.path.exists(path):
            # use_column_width, not use_container_width: st.image only gained the
            # latter in streamlit 1.40 and requirements.txt pins 1.39.
            col.image(path, caption=cap, use_column_width=True)
