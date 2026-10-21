"""Investigator-facing output: plain-language reasons per account and ring reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

DAY = 86_400


def reasons(feats: pd.DataFrame, account: int, flagged: set[int] | None = None, txns: pd.DataFrame | None = None) -> list[str]:
    """Behavioural red flags, each compared with the population."""
    f = feats.loc[account]
    pct = lambda col: float((feats[col] < f[col]).mean())
    out = []
    if f["passthrough"] > 0.85 and f["amt_in"] > 0 and f["median_hold_min"] < 180:
        out.append(f"money out equals {f['passthrough']:.0%} of money in; payments leave a median {f['median_hold_min']:.0f} min after funds arrive")
    elif f["fast_forward_share"] > 0.3:
        out.append(f"{f['fast_forward_share']:.0%} of outgoing money leaves within an hour of an incoming payment")
    if f["uniq_senders"] >= 5 and f["one_off_sender_share"] > 0.6:
        out.append(f"{f['uniq_senders']:.0f} different senders, {f['one_off_sender_share']:.0%} of whom paid only once")
    if f["age_days"] < 90:
        out.append(f"account opened {f['age_days']:.0f} days ago")
    if f["amt_in"] > 0 and pct("amt_in") > 0.97:
        out.append(f"received ₹{f['amt_in']:,.0f}, more than {pct('amt_in'):.0%} of customers")
    if f["structuring_share"] > 0.1:
        out.append(f"{f['structuring_share']:.0%} of transfers sit just under the ₹50,000 reporting threshold")
    if f["cash_out_share"] > 0.6:
        out.append(f"{f['cash_out_share']:.0%} of outflow is cash or crypto")
    if f["device_peers"] >= 2:
        out.append(f"device shared with {f['device_peers']:.0f} other accounts")
    if f["night_share"] > 0.2:
        out.append(f"{f['night_share']:.0%} of activity between midnight and 5 am")
    if flagged and txns is not None:
        nb = set(txns.loc[txns["src"] == account, "dst"]) | set(txns.loc[txns["dst"] == account, "src"])
        linked = len(nb & flagged - {account})
        if linked:
            out.append(f"directly transacts with {linked} other flagged account{'s' if linked > 1 else ''}")
    return out


@dataclass
class Ring:
    ring_id: int
    members: list[int]
    entry: list[int]
    exits: list[int]
    victims: list[int]
    victim_inflow: float
    cashed_out: float
    first_ts: int
    last_ts: int
    shared_devices: int
    score: float
    flows: pd.DataFrame = field(repr=False, default=None)


def find_rings(accounts: pd.DataFrame, txns: pd.DataFrame, scores: np.ndarray, threshold: float, min_size: int = 3) -> list[Ring]:
    """Connect flagged accounts through money flows and shared devices; each
    connected component with at least `min_size` members is a ring candidate."""
    flagged = set(np.flatnonzero(scores >= threshold))
    kind = accounts.set_index("account")["kind"]
    sinks = set(kind[kind.isin(["atm", "exchange"])].index)

    g = nx.Graph()
    g.add_nodes_from(flagged)

    def link(a, b, w):
        if a != b:
            g.add_edge(a, b, weight=g.get_edge_data(a, b, {"weight": 0})["weight"] + w)

    inner = txns[txns["src"].isin(flagged) & txns["dst"].isin(flagged)]
    for (a, b), amt in inner.groupby(["src", "dst"])["amount"].sum().items():
        link(a, b, np.log1p(amt))
    dev = accounts.loc[accounts["account"].isin(flagged) & (accounts["device"] != ""), ["account", "device"]]
    for _, grp in dev.groupby("device"):
        ids = grp["account"].tolist()
        for a, b in zip(ids, ids[1:]):
            link(a, b, 5.0)
    # two flagged accounts paid by the same outsider (a shared victim) are weakly linked
    into = txns[txns["dst"].isin(flagged) & ~txns["src"].isin(flagged | sinks)]
    for _, grp in into.groupby("src")["dst"]:
        ids = grp.unique().tolist()
        for a, b in zip(ids, ids[1:]):
            link(a, b, 1.0)

    # hubs (false positives with many links) would glue rings together; communities split them
    comps = []
    for comp in nx.connected_components(g):
        if len(comp) <= 25:
            comps.append(comp)
        else:
            comps.extend(nx.community.louvain_communities(g.subgraph(comp), weight="weight", seed=0))

    rings = []
    for comp in comps:
        if len(comp) < min_size:
            continue
        members = sorted(comp)
        m = set(members)
        touching = txns[txns["src"].isin(m) | txns["dst"].isin(m)]
        incoming = touching[touching["dst"].isin(m) & ~touching["src"].isin(m)]
        outgoing = touching[touching["src"].isin(m) & ~touching["dst"].isin(m)]
        # victims: outsiders paying in who are not salary payers / settlement nodes
        payers = incoming[~incoming["src"].map(kind).isin(["employer", "external"])]
        cash = outgoing[outgoing["dst"].isin(sinks)]
        devs = accounts.loc[accounts["account"].isin(m), "device"]
        rings.append(Ring(
            ring_id=len(rings),
            members=members,
            entry=sorted(set(payers["dst"])),
            exits=sorted(set(cash["src"])),
            victims=sorted(set(payers["src"])),
            victim_inflow=float(payers["amount"].sum()),
            cashed_out=float(cash["amount"].sum()),
            first_ts=int(touching["ts"].min()),
            last_ts=int(touching["ts"].max()),
            shared_devices=int((devs.value_counts() > 1).sum()),
            score=float(np.mean(scores[members])),
            flows=touching,
        ))
    return sorted(rings, key=lambda r: -r.victim_inflow)


def ring_recovery(accounts: pd.DataFrame, rings: list[Ring], subset: np.ndarray | None = None) -> pd.DataFrame:
    """For each true ring, best Jaccard overlap with a detected ring."""
    truth = accounts[accounts["ring_id"] >= 0]
    if subset is not None:
        truth = truth[subset[truth.index]]
    rows = []
    for rid, grp in truth.groupby("ring_id"):
        t = set(grp["account"])
        best = max((len(t & set(r.members)) / len(t | set(r.members)) for r in rings), default=0.0)
        found = max((len(t & set(r.members)) for r in rings), default=0)
        rows.append({"true_ring": rid, "size": len(t), "members_found": found, "jaccard": best})
    return pd.DataFrame(rows)


def ring_roles(r: Ring, accounts: pd.DataFrame) -> dict[int, str]:
    """Infer each member's position in the laundering chain from the flows."""
    roles = {}
    for a in r.members:
        receives_out = a in r.entry
        cashes = a in r.exits
        roles[a] = "entry" if receives_out and not cashes else ("cash-out" if cashes else "layer")
    return roles
