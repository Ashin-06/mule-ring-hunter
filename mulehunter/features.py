"""Behavioural features per account, built from the raw transaction log."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400
FAST_FORWARD_S = 3600
STRUCT_BAND = (40_000, 50_000)

FEATURES = [
    "age_days", "n_in", "n_out", "amt_in", "amt_out", "passthrough",
    "uniq_senders", "uniq_receivers", "one_off_sender_share", "median_hold_min", "fast_forward_share",
    "night_share", "structuring_share", "cash_out_share", "device_peers", "burstiness", "in_amount_cv",
]
LOG_FEATURES = {"age_days", "n_in", "n_out", "amt_in", "amt_out",
                "uniq_senders", "uniq_receivers", "median_hold_min", "device_peers", "burstiness"}


def account_features(accounts: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    now = int(txns["ts"].max())
    idx = accounts["account"].to_numpy()
    f = pd.DataFrame(index=pd.Index(idx, name="account"))
    kind = accounts.set_index("account")["kind"]
    sinks = set(kind[kind.isin(["atm", "exchange"])].index)

    created = accounts.set_index("account")["created_ts"]
    f["age_days"] = ((now - created) / DAY).clip(lower=0)

    out = txns.rename(columns={"src": "account", "dst": "peer"})
    inc = txns.rename(columns={"dst": "account", "src": "peer"})
    go, gi = out.groupby("account"), inc.groupby("account")
    f["n_in"] = gi.size()
    f["n_out"] = go.size()
    f["amt_in"] = gi["amount"].sum()
    f["amt_out"] = go["amount"].sum()
    f = f.fillna(0.0)
    f["passthrough"] = (f["amt_out"] / (f["amt_in"] + 1.0)).clip(0, 3)
    f["uniq_senders"] = gi["peer"].nunique()
    f["uniq_receivers"] = go["peer"].nunique()

    pair_counts = inc.groupby(["account", "peer"]).size()
    f["one_off_sender_share"] = (pair_counts == 1).groupby(level=0).mean()

    both = pd.concat([out[["account", "ts"]], inc[["account", "ts"]]])
    per_day = both.assign(day=both["ts"] // DAY).groupby(["account", "day"]).size()
    f["burstiness"] = per_day.groupby(level=0).max() / per_day.groupby(level=0).mean()
    hours = (both["ts"] % DAY) // 3600
    f["night_share"] = (hours < 5).groupby(both["account"]).mean()

    # holding time: for every outgoing payment, time since the latest incoming one
    o = out[["account", "ts", "amount"]].sort_values("ts")
    i = inc[["account", "ts"]].rename(columns={"ts": "in_ts"}).sort_values("in_ts")
    m = pd.merge_asof(o, i, left_on="ts", right_on="in_ts", by="account", direction="backward")
    m["hold"] = m["ts"] - m["in_ts"]
    f["median_hold_min"] = m.groupby("account")["hold"].median() / 60
    fast = m[m["hold"] <= FAST_FORWARD_S].groupby("account")["amount"].sum()
    f["fast_forward_share"] = fast / f["amt_out"].replace(0, np.nan)

    band = out["amount"].between(*STRUCT_BAND, inclusive="left")
    f["structuring_share"] = band.groupby(out["account"]).mean()
    cash = out[out["peer"].isin(sinks)].groupby("account")["amount"].sum()
    f["cash_out_share"] = cash / f["amt_out"].replace(0, np.nan)

    dev = accounts.set_index("account")["device"]
    f["device_peers"] = dev.map(dev.value_counts()).fillna(1) - 1
    f.loc[dev == "", "device_peers"] = 0

    f["in_amount_cv"] = gi["amount"].std() / gi["amount"].mean()

    defaults = {"median_hold_min": 30 * 24 * 60.0}
    for c in FEATURES:
        f[c] = f[c].astype(float).fillna(defaults.get(c, 0.0))
    return f[FEATURES]


def model_matrix(f: pd.DataFrame) -> np.ndarray:
    """log-scale heavy-tailed counts/amounts, then standardise."""
    x = f.copy()
    for c in LOG_FEATURES:
        x[c] = np.log1p(x[c].clip(lower=0))
    x = (x - x.mean()) / (x.std() + 1e-9)
    return x.to_numpy(dtype=np.float32)
