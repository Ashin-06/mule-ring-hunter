"""Synthetic instant-payment network with embedded money-mule rings.

Normal life: salaries, merchant spend, bills, P2P between friends, ATM cash, a little
crypto. It also includes hard negatives that look like mules at first glance:
merchants settling daily (fan-in and pass-through), group collectors who pool money
from friends and forward it (chit funds, trip money) and landlords.

Mule rings follow the layering pattern seen in scam-proceeds laundering:
victims -> tier-1 receivers -> tier-2 layering -> cash-out (ATM / crypto),
with fast forwarding, structuring below reporting limits, device sharing and burst
account creation. About a third of tier-1 mules are rented old accounts, so account
age alone does not give them away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86_400
STRUCTURING_LIMIT = 50_000  # transfers are split to stay below this
ATM_LIMIT = 10_000


@dataclass
class SimConfig:
    n_retail: int = 6000
    n_merchants: int = 250
    n_employers: int = 50
    n_exchanges: int = 4
    n_collectors: int = 60
    n_landlords: int = 80
    n_traders: int = 150
    n_rings: int = 16
    days: int = 45
    seed: int = 7


class _Ledger:
    def __init__(self):
        self.cols = {k: [] for k in ("ts", "src", "dst", "amount", "channel")}

    def add(self, ts, src, dst, amount, channel):
        ts, src, dst, amount = map(np.atleast_1d, (ts, src, dst, amount))
        if min(len(ts), len(src), len(dst), len(amount)) == 0:
            return
        n = max(len(ts), len(src), len(dst), len(amount))
        for k, v in (("ts", ts), ("src", src), ("dst", dst), ("amount", amount)):
            self.cols[k].append(np.broadcast_to(v, n))
        self.cols["channel"].append(np.full(n, channel))

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame({k: np.concatenate(v) for k, v in self.cols.items()})
        df["ts"] = df["ts"].astype(np.int64)
        df["amount"] = df["amount"].round(2)
        df = df[df["amount"] > 0].sort_values("ts").reset_index(drop=True)
        df.insert(0, "txn_id", np.arange(len(df)))
        return df


def _daytime(rng, n, days):
    """Timestamps biased to waking hours."""
    day = rng.integers(0, days, n)
    hour = rng.normal(14.5, 4.5, n) % 24  # mostly daytime, some late-night activity
    return day * DAY + (hour * 3600).astype(int) + rng.integers(0, 60, n)


def simulate(cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    T = cfg.days * DAY
    led = _Ledger()

    rows = []

    def new_accounts(kind, n, created, ring=-1, label=0, role=""):
        start = len(rows)
        for i in range(n):
            rows.append({"account": start + i, "kind": kind, "created_ts": int(created[i]), "ring_id": ring,
                         "is_mule": label, "role": role, "device": f"D{len(rows)}", "rented": False,
                         "ring_style": "", "ring_start": -1})
        return np.arange(start, start + n)

    old = lambda n: rng.integers(-4 * 365 * DAY, -90 * DAY, n)
    retail = new_accounts("retail", cfg.n_retail, rng.integers(-5 * 365 * DAY, int(T * 0.4), cfg.n_retail))
    merchants = new_accounts("merchant", cfg.n_merchants, old(cfg.n_merchants))
    employers = new_accounts("employer", cfg.n_employers, old(cfg.n_employers))
    exchanges = new_accounts("exchange", cfg.n_exchanges, old(cfg.n_exchanges))
    ext = new_accounts("external", 1, old(1))[0]  # other banks / settlement
    atm = new_accounts("atm", 1, old(1))[0]  # cash leaves the system
    utilities = merchants[: max(5, cfg.n_merchants // 25)]

    # households share devices occasionally (benign device sharing)
    for a in rng.choice(retail, cfg.n_retail // 8, replace=False):
        rows[a]["device"] = rows[int(rng.choice(retail))]["device"]

    # ---- salaries (monthly) and employer funding
    employer_of = rng.choice(employers, len(retail))
    salary = np.exp(rng.normal(10.4, 0.55, len(retail)))
    for payday in range(0, cfg.days, 30):
        ts = payday * DAY + 10 * 3600 + rng.integers(0, 7200, len(retail))
        led.add(ts, employer_of, retail, salary, "NEFT")
        for e in employers:
            led.add(payday * DAY + 8 * 3600, ext, e, salary[employer_of == e].sum() * 1.02 + 1, "NEFT")

    # ---- merchant spend, bills, P2P, cash, crypto
    pref = rng.choice(merchants, (len(retail), 8))
    n_spend = rng.poisson(0.7 * cfg.days, len(retail))
    who = np.repeat(np.arange(len(retail)), n_spend)
    led.add(_daytime(rng, len(who), cfg.days), retail[who], pref[who, rng.integers(0, 8, len(who))],
            np.exp(rng.normal(6.4, 1.0, len(who))), "UPI")

    led.add(_daytime(rng, len(retail), cfg.days), retail, rng.choice(utilities, len(retail)),
            np.exp(rng.normal(7.2, 0.5, len(retail))), "UPI")

    friends = rng.choice(retail, (len(retail), 6))
    n_p2p = rng.poisson(0.12 * cfg.days, len(retail))
    who = np.repeat(np.arange(len(retail)), n_p2p)
    led.add(_daytime(rng, len(who), cfg.days), retail[who], friends[who, rng.integers(0, 6, len(who))],
            np.exp(rng.normal(7.3, 1.1, len(who))), "UPI")

    n_atm = rng.poisson(0.08 * cfg.days, len(retail))
    who = np.repeat(np.arange(len(retail)), n_atm)
    led.add(_daytime(rng, len(who), cfg.days), retail[who], atm,
            np.round(np.exp(rng.normal(8.2, 0.5, len(who))), -2), "ATM")

    crypto_users = rng.choice(retail, len(retail) // 25, replace=False)
    for a in crypto_users:
        k = rng.integers(1, 5)
        led.add(_daytime(rng, k, cfg.days), a, rng.choice(exchanges, k), np.exp(rng.normal(9.0, 0.8, k)), "IMPS")

    # ---- group collectors: friends chip in, collector forwards within an hour (hard negative)
    collectors = rng.choice(retail, cfg.n_collectors, replace=False)
    for c in collectors:
        circle = rng.choice(retail, rng.integers(8, 20), replace=False)
        payee = rng.choice(np.concatenate([merchants, retail]))
        for event_day in rng.choice(cfg.days, rng.integers(1, 3), replace=False):
            t0 = event_day * DAY + 11 * 3600
            amt = rng.uniform(500, 3000)
            ts = t0 + rng.integers(0, 6 * 3600, len(circle))
            led.add(ts, circle, c, amt * rng.uniform(0.9, 1.1, len(circle)), "UPI")
            led.add(ts.max() + rng.integers(600, 3600), c, payee, amt * len(circle) * 0.98, "IMPS")

    # ---- landlords: rent from tenants, mortgage out
    for l in rng.choice(retail, cfg.n_landlords, replace=False):
        tenants = rng.choice(retail, rng.integers(2, 6), replace=False)
        rent = rng.uniform(8_000, 25_000)
        for d0 in range(2, cfg.days, 30):
            led.add(d0 * DAY + rng.integers(0, 3 * DAY, len(tenants)), tenants, l, rent, "UPI")
            led.add((d0 + 4) * DAY, l, ext, rent * len(tenants) * 0.7, "NEFT")

    # ---- small traders / resellers: many one-off buyers, supplier paid within hours (hard negative)
    for tr in rng.choice(retail, cfg.n_traders, replace=False):
        supplier = rng.choice(np.concatenate([merchants, retail]))
        for d in rng.choice(cfg.days, rng.integers(4, 12), replace=False):
            buyers = rng.choice(retail, rng.integers(3, 10), replace=False)
            ts = d * DAY + 9 * 3600 + rng.integers(0, 10 * 3600, len(buyers))
            amts = np.exp(rng.normal(8.3, 0.7, len(buyers)))
            led.add(ts, buyers, tr, amts, "UPI")
            led.add(ts.max() + rng.integers(1800, 5 * 3600), tr, supplier, amts.sum() * rng.uniform(0.85, 0.95), "IMPS")

    # ---- mule rings
    taken = set(collectors) | set(crypto_users)
    victims_pool = np.setdiff1d(retail, list(taken))
    for ring in range(cfg.n_rings):
        start = int(rng.integers(3, cfg.days - 12)) * DAY
        end = start + int(rng.integers(4, 12)) * DAY
        burst = start - int(rng.integers(5, 40)) * DAY
        sizes = rng.integers(2, 6), rng.integers(1, 4), rng.integers(1, 3)
        careful = rng.random() < 0.4  # slower forwarding, keeps a cut, no shared phones

        def ring_accounts(n, role):
            # rented accounts: real customers with a normal history who let the ring use them
            rented = rng.random(n) < {"tier1": 0.5, "cashout": 0.3}.get(role, 0.0)
            k = int((~rented).sum())
            fresh = new_accounts("retail", k, burst + rng.integers(0, 5 * DAY, k))
            hosts = rng.choice(victims_pool, int(rented.sum()), replace=False)
            ids = np.concatenate([fresh, hosts]).astype(int)
            for acct, r in zip(ids, np.concatenate([np.zeros(k, bool), np.ones(len(hosts), bool)])):
                rows[acct].update(ring_id=ring, is_mule=1, role=role, rented=bool(r),
                                  ring_style="careful" if careful else "fast", ring_start=start)
            return ids

        t1, t2, co = ring_accounts(sizes[0], "tier1"), ring_accounts(sizes[1], "tier2"), ring_accounts(sizes[2], "cashout")
        members = np.concatenate([t1, t2, co])
        victims_pool = np.setdiff1d(victims_pool, members)
        if not careful:
            devices = [f"RD{ring}_{i}" for i in range(max(1, len(members) // 3))]
            for m in members:
                rows[m]["device"] = str(rng.choice(devices))

        # camouflage: ordinary spend on the fresh accounts
        for m in members:
            k = rng.integers(2, 12)
            led.add(_daytime(rng, k, cfg.days), m, rng.choice(merchants, k), np.exp(rng.normal(6.3, 0.9, k)), "UPI")

        hold1, hold2, keep = (6 * 3600, 10 * 3600, (0.75, 0.9)) if careful else (25 * 60, 40 * 60, (0.92, 0.985))
        victims = rng.choice(victims_pool, rng.integers(6, 45), replace=False)
        for v in victims:
            for _ in range(rng.integers(1, 3)):
                ts = int(start + _daytime(rng, 1, (end - start) // DAY)[0])  # victims pay during the day
                amt = float(np.exp(rng.normal(8.7, 1.0)))
                hop1 = int(rng.choice(t1))
                led.add(ts, v, hop1, amt, "UPI")
                fwd = amt * rng.uniform(*keep)
                t = ts + int(rng.exponential(hold1)) + 60
                for chunk in _split(fwd, rng):
                    hop2 = int(rng.choice(t2))
                    led.add(t, hop1, hop2, chunk, "IMPS")
                    t2_time = t + int(rng.exponential(hold2)) + 60
                    out = int(rng.choice(co))
                    led.add(t2_time, hop2, out, chunk * rng.uniform(0.95, 0.99), "IMPS")
                    _cash_out(led, rng, out, chunk * 0.95, t2_time, atm, exchanges)
                    t += rng.integers(30, 300)

    # ---- merchants settle the day's takings each night: fan-in + ~97% pass-through (hard negative)
    pre = led.frame()
    takings = pre[pre["dst"].isin(merchants)].groupby([pre["dst"], pre["ts"] // DAY])["amount"].sum()
    m_ids = takings.index.get_level_values(0).to_numpy()
    days = takings.index.get_level_values(1).to_numpy()
    led.add(days * DAY + 23 * 3600 + 1800 + rng.integers(0, 600, len(days)), m_ids, ext,
            takings.to_numpy() * rng.uniform(0.95, 0.99, len(days)), "NEFT")

    accounts = pd.DataFrame(rows)
    txns = led.frame()
    txns = txns[(txns["ts"] >= 0) & (txns["ts"] < T + 2 * DAY)].reset_index(drop=True)
    return accounts, txns


def _split(amount: float, rng) -> list[float]:
    if amount < STRUCTURING_LIMIT:
        return [amount]
    k = int(np.ceil(amount / (STRUCTURING_LIMIT * rng.uniform(0.8, 0.98))))
    w = rng.dirichlet(np.ones(k) * 8)
    return list(amount * w)


def _cash_out(led, rng, acct, amount, t, atm, exchanges):
    t += int(rng.uniform(1, 6) * 3600)
    if rng.random() < 0.7:
        while amount > 500:
            w = min(amount, ATM_LIMIT * rng.uniform(0.6, 1.0))
            led.add(t, acct, atm, round(w, -2), "ATM")
            amount -= w
            t += int(rng.integers(120, 900))
    else:
        led.add(t, acct, rng.choice(exchanges), amount, "IMPS")


# --------------------------------------------------------------- real datasets

def load_ibm_aml(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IBM AMLworld transactions CSV (HI-Small_Trans.csv etc.). Accounts touching a
    laundering transaction are labelled positive."""
    df = pd.read_csv(path)
    src = df["From Bank"].astype(str) + ":" + df["Account"].astype(str)
    dst = df["To Bank"].astype(str) + ":" + df["Account.1"].astype(str)
    ids = pd.Index(pd.unique(pd.concat([src, dst])))
    ts = pd.to_datetime(df["Timestamp"]).astype("int64") // 10**9
    txns = pd.DataFrame({
        "txn_id": np.arange(len(df)), "ts": ts - ts.min(), "src": ids.get_indexer(src), "dst": ids.get_indexer(dst),
        "amount": df["Amount Paid"].astype(float), "channel": df["Payment Format"].astype(str),
    }).sort_values("ts").reset_index(drop=True)
    bad = df["Is Laundering"] == 1
    pos = set(ids.get_indexer(src[bad])) | set(ids.get_indexer(dst[bad]))
    accounts = pd.DataFrame({"account": np.arange(len(ids)), "kind": "retail", "created_ts": 0, "ring_id": -1,
                             "is_mule": [int(i in pos) for i in range(len(ids))], "role": "", "device": ""})
    return accounts, txns


def load_paysim(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PaySim CSV. Destinations of fraudulent TRANSFERs are the mule accounts."""
    df = pd.read_csv(path)
    ids = pd.Index(pd.unique(pd.concat([df["nameOrig"], df["nameDest"]])))
    txns = pd.DataFrame({
        "txn_id": np.arange(len(df)), "ts": df["step"].astype(np.int64) * 3600, "src": ids.get_indexer(df["nameOrig"]),
        "dst": ids.get_indexer(df["nameDest"]), "amount": df["amount"].astype(float), "channel": df["type"].astype(str),
    })
    mules = set(ids.get_indexer(df.loc[(df["isFraud"] == 1) & (df["type"] == "TRANSFER"), "nameDest"]))
    accounts = pd.DataFrame({"account": np.arange(len(ids)), "kind": "retail", "created_ts": 0, "ring_id": -1,
                             "is_mule": [int(i in mules) for i in range(len(ids))], "role": "", "device": ""})
    return accounts, txns
