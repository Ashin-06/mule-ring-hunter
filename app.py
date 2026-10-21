"""Investigation dashboard.

    python -m mulehunter train      # once, produces artifacts/
    streamlit run app.py
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mulehunter.explain import find_rings, reasons, ring_recovery, ring_roles

DATA, ART = Path("data"), Path("artifacts")
st.set_page_config(page_title="Mule Ring Hunter", layout="wide")
st.title("Mule Ring Hunter")

if not (ART / "scores.csv").exists():
    st.info("No trained artifacts yet. Run `python -m mulehunter train` first, then reload this page.")
    st.stop()


@st.cache_data
def load():
    a = pd.read_csv(DATA / "accounts.csv.gz", keep_default_na=False)
    t = pd.read_csv(DATA / "txns.csv.gz")
    s = pd.read_csv(ART / "scores.csv")
    f = pd.read_csv(ART / "features.csv").set_index("account")
    run = json.load(open(ART / "run.json"))
    m = pd.read_json(ART / "metrics.json")
    return a, t, s, f, run, m


accounts, txns, scores, feats, run, metrics = load()
score = scores["score"].to_numpy()
kind = accounts.set_index("account")["kind"]


@st.cache_data
def rings_at(threshold):
    return find_rings(accounts, txns, score, threshold)


budget = st.sidebar.slider("Alert budget (share of customers reviewed)", 0.005, 0.05, float(run["budget"]), 0.005, format="%.3f")
eligible = accounts["kind"].isin(["retail", "merchant"]).to_numpy()
thr = float(np.quantile(score[eligible], 1 - budget))
rings = rings_at(thr)
flagged = set(np.flatnonzero(score >= thr))
st.sidebar.metric("Accounts flagged", len(flagged))
st.sidebar.metric("Ring candidates", len(rings))
st.sidebar.metric("Victim money in rings", f"₹{sum(r.victim_inflow for r in rings):,.0f}")

tab_over, tab_rings, tab_acct = st.tabs(["Model", "Rings", "Account lookup"])

with tab_over:
    st.subheader("Held-out rings (never seen in training)")
    st.dataframe(metrics.round(3), width="stretch")
    if "ring_id" in accounts and (accounts["ring_id"] >= 0).any():
        rec = ring_recovery(accounts, rings)
        c1, c2 = st.columns(2)
        c1.metric("True rings recovered (Jaccard ≥ 0.5)", f"{(rec['jaccard'] >= 0.5).mean():.0%}")
        c2.metric("Mean Jaccard", f"{rec['jaccard'].mean():.2f}")
    if (ART / "timeline.csv").exists():
        tl = pd.read_csv(ART / "timeline.csv")
        fig = go.Figure([go.Scatter(x=tl["day"], y=tl[c], mode="lines+markers", name=c.replace(" PR-AUC", ""))
                         for c in tl.columns if c.endswith("PR-AUC")])
        fig.update_layout(title="Early detection: PR-AUC using only the first N days", xaxis_title="day", yaxis_title="PR-AUC", height=360)
        st.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Histogram(x=score[eligible], nbinsx=60))
    fig.add_vline(x=thr, line_dash="dot", annotation_text="alert threshold")
    fig.update_layout(title="Score distribution", yaxis_type="log", height=300)
    st.plotly_chart(fig, width="stretch")

with tab_rings:
    if not rings:
        st.warning("No ring candidates at this budget.")
    else:
        table = pd.DataFrame([{
            "ring": r.ring_id, "members": len(r.members), "victims": len(r.victims),
            "victim inflow ₹": round(r.victim_inflow), "cashed out ₹": round(r.cashed_out),
            "active days": round((r.last_ts - r.first_ts) / 86400, 1), "shared devices": r.shared_devices,
            "mean score": round(r.score, 3),
        } for r in rings])
        st.dataframe(table, hide_index=True, width="stretch")
        rid = st.selectbox("Open ring", table["ring"])
        r = next(x for x in rings if x.ring_id == rid)
        roles = ring_roles(r, accounts)

        # money-flow graph: members, victims, cash sinks
        g = nx.DiGraph()
        flows = r.flows.groupby(["src", "dst"])["amount"].sum().reset_index()
        keep = set(r.members) | set(r.victims[:40]) | {a for a in flows["dst"] if kind.get(a) in ("atm", "exchange")}
        for _, e in flows.iterrows():
            if e["src"] in keep and e["dst"] in keep:
                g.add_edge(int(e["src"]), int(e["dst"]), amount=e["amount"])
        pos = nx.spring_layout(g, seed=3, k=0.6)
        palette = {"entry": "#e4572e", "layer": "#f3a712", "cash-out": "#a8201a", "victim": "#8d99ae", "sink": "#2b2d42"}

        def role(n):
            return roles.get(n) or ("sink" if kind.get(n) in ("atm", "exchange") else "victim")

        ex, ey = [], []
        for a, b in g.edges():
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]
        fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#b0b7c3"), hoverinfo="skip"))
        for rl, color in palette.items():
            nodes = [n for n in g.nodes if role(n) == rl]
            if nodes:
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=rl,
                    marker=dict(size=[8 if rl == "victim" else 16 for _ in nodes], color=color, line=dict(width=1, color="white")),
                    text=[f"account {n}<br>{rl}<br>score {score[n]:.3f}" for n in nodes], hoverinfo="text"))
        fig.update_layout(height=520, xaxis_visible=False, yaxis_visible=False, title="Money flow inside the ring")
        st.plotly_chart(fig, width="stretch")

        # Sankey: victims -> entry -> layer -> cash-out -> sinks
        stage = lambda n: {"entry": 1, "layer": 2, "cash-out": 3}.get(roles.get(n), 0 if kind.get(n) not in ("atm", "exchange") else 4)
        names = ["victims", "entry mules", "layering mules", "cash-out mules", "ATM / crypto"]
        agg = {}
        for _, e in flows.iterrows():
            s, d = stage(int(e["src"])), stage(int(e["dst"]))
            if s < d:
                agg[(s, d)] = agg.get((s, d), 0) + e["amount"]
        if agg:
            fig = go.Figure(go.Sankey(node=dict(label=names, pad=20),
                                      link=dict(source=[k[0] for k in agg], target=[k[1] for k in agg], value=list(agg.values()))))
            fig.update_layout(height=320, title="Where the money went")
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Members**")
        for m in r.members:
            with st.expander(f"account {m} · {roles[m]} · score {score[m]:.3f}"):
                for line in reasons(feats, m, flagged, txns):
                    st.write("•", line)

with tab_acct:
    acct = st.number_input("Account id", 0, int(accounts["account"].max()), int(next(iter(flagged), 0)))
    s = scores.set_index("account").loc[acct]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble score", f"{s['score']:.3f}", "flagged" if acct in flagged else "not flagged", delta_color="inverse" if acct in flagged else "off")
    c2.metric("GraphSAGE", f"{s['directed GraphSAGE']:.3f}")
    c3.metric("XGBoost", f"{s['features + XGBoost']:.3f}")
    for line in reasons(feats, acct, flagged, txns):
        st.write("•", line)
    hist = txns[(txns["src"] == acct) | (txns["dst"] == acct)].copy()
    hist["direction"] = np.where(hist["src"] == acct, "out", "in")
    hist["counterparty"] = np.where(hist["src"] == acct, hist["dst"], hist["src"])
    hist["time"] = pd.to_datetime(hist["ts"], unit="s", origin=pd.Timestamp("2025-01-01"))
    st.dataframe(hist[["time", "direction", "counterparty", "amount", "channel"]].sort_values("time"), hide_index=True, width="stretch")
