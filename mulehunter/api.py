"""Analysis service: finds likely mule accounts and ring candidates in a batch of transactions.

    POST /v1/analyse          accounts + transactions -> scored accounts, flags within an alert budget, rings [analyse]
    GET  /v1/analyses[/{id}]  past analyses and their findings [read]
    GET  /healthz, /readyz, /metrics

The models are the ones `python -m mulehunter train` writes: gnn.pt (a torch state dict, loaded with
weights_only=True) and xgb.json. Neither format runs code when loaded. Features are standardised
within each batch, so a batch should be a meaningful slice of the payment graph (a day or a
week of a region), not a handful of accounts. Settings: ops.py and docs/runbook.md.

    python -m mulehunter serve
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from xgboost import XGBClassifier

from . import db, ops
from .explain import find_rings
from .features import account_features, model_matrix
from .model import SCORED_KINDS, DirSAGE, ensemble

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
MODEL_DIR = Path(os.environ.get("MULEHUNTER_MODEL_DIR", "").strip() or "artifacts")
if settings.production and not os.environ.get("MULEHUNTER_MODEL_DIR", "").strip():
    raise ops.ConfigError("production needs MULEHUNTER_MODEL_DIR: gnn.pt and xgb.json from `python -m mulehunter train`")
MAX_ACCOUNTS, MAX_TXNS = 20_000, 50_000


def _load_models():
    try:
        xgb = XGBClassifier()
        xgb.load_model(MODEL_DIR / "xgb.json")
        state = torch.load(MODEL_DIR / "gnn.pt", weights_only=True)  # tensors only: no code runs
        gnn = DirSAGE(state["inp.weight"].shape[1])
        gnn.load_state_dict(state)
        gnn.eval()
    except (OSError, KeyError, RuntimeError, ValueError) as e:
        raise ops.ConfigError(f"cannot load models from {MODEL_DIR}: {type(e).__name__}") from None
    return gnn, xgb


GNN, XGB = _load_models()
engine = db.connect(settings.database_url)
_lock = threading.Lock()

app = FastAPI(title="Mule ring hunter", version="1.0.0")
ops.install(app, settings, {"database": lambda: engine.connect().close(), "models": lambda: GNN is not None})


class Account(BaseModel):
    account: str = Field(..., min_length=1, max_length=64)
    kind: Literal["retail", "merchant", "employer", "exchange", "external", "atm"]
    created_ts: int = Field(..., description="account opening time, seconds on the same clock as ts")
    device: str = Field("", max_length=64)


class Txn(BaseModel):
    ts: int = Field(..., ge=0)
    src: str = Field(..., min_length=1, max_length=64)
    dst: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., gt=0, le=1e10, allow_inf_nan=False)
    channel: str = Field("UPI", max_length=16)


class AnalyseRequest(BaseModel):
    accounts: List[Account] = Field(..., min_length=3, max_length=MAX_ACCOUNTS)
    txns: List[Txn] = Field(..., min_length=1, max_length=MAX_TXNS)
    alert_budget: float = Field(0.01, gt=0, le=0.2, description="share of retail and merchant accounts to flag")
    label: str = Field("", max_length=200)


def score_batch(accounts: pd.DataFrame, txns: pd.DataFrame) -> np.ndarray:
    """Ensemble score per account (0 for kinds that aren't scored), accounts numbered 0..n-1."""
    feats = account_features(accounts, txns)
    x = torch.as_tensor(model_matrix(feats))
    pairs = txns.groupby(["src", "dst"]).size().reset_index()[["src", "dst"]].to_numpy().T
    with torch.no_grad():
        p_gnn = torch.sigmoid(GNN(x, torch.as_tensor(pairs, dtype=torch.long))).numpy()
    p_xgb = XGB.predict_proba(feats.to_numpy())[:, 1]
    eligible = accounts["kind"].isin(SCORED_KINDS).to_numpy()
    p = ensemble(p_gnn, p_xgb, eligible)
    p[~eligible] = 0.0
    return p


@app.post("/v1/analyse")
def analyse(req: AnalyseRequest, ident: dict = Depends(ops.require("analyse"))) -> Dict[str, Any]:
    ids = [a.account for a in req.accounts]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "account IDs must be unique")
    index = {a: i for i, a in enumerate(ids)}
    if any(t.src not in index or t.dst not in index for t in req.txns):
        raise HTTPException(422, "every transaction's src and dst must be in accounts")
    accounts = pd.DataFrame([{"account": index[a.account], "kind": a.kind, "created_ts": a.created_ts, "device": a.device}
                             for a in req.accounts])
    txns = pd.DataFrame([{"txn_id": i, "ts": t.ts, "src": index[t.src], "dst": index[t.dst], "amount": t.amount,
                          "channel": t.channel} for i, t in enumerate(req.txns)])
    p = score_batch(accounts, txns)
    eligible = accounts["kind"].isin(SCORED_KINDS).to_numpy()
    if not eligible.any():
        raise HTTPException(422, "no retail or merchant accounts to score")
    thr = float(np.quantile(p[eligible], 1 - req.alert_budget))
    flagged = np.flatnonzero(eligible & (p >= thr) & (p > 0))
    rings = find_rings(accounts, txns, p, thr)
    findings = {
        "threshold": thr,
        "flagged": [{"account": ids[i], "score": round(float(p[i]), 6)} for i in flagged[np.argsort(-p[flagged])]],
        "rings": [{**{k: [ids[int(m)] for m in getattr(r, k)] for k in ("members", "entry", "exits", "victims")},
                   **{k: round(float(getattr(r, k)), 2) for k in ("victim_inflow", "cashed_out", "score")},
                   **{k: int(getattr(r, k)) for k in ("first_ts", "last_ts", "shared_devices")}} for r in rings],
    }
    aid = uuid.uuid4().hex
    with _lock, engine.begin() as c:
        c.execute(text("INSERT INTO analyses VALUES (:i, :t, :u, :l, :a, :x, :f, :r, :fd)"),
                  {"i": aid, "t": datetime.now(timezone.utc).isoformat(timespec="microseconds"), "u": ident["user"] or "demo",
                   "l": req.label, "a": len(ids), "x": len(req.txns), "f": len(flagged), "r": len(rings),
                   "fd": json.dumps(findings)})
    ops.DECISIONS.labels(decision="accounts_flagged").inc(len(flagged))
    ops.DECISIONS.labels(decision="rings_found").inc(len(rings))
    return {"analysis_id": aid, **findings}


@app.get("/v1/analyses", dependencies=[Depends(ops.require("read"))])
def list_analyses(limit: int = Query(50, ge=1, le=1000)) -> Dict[str, Any]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT id, created_at, requested_by, label, accounts, transactions, flagged, rings "
                              "FROM analyses ORDER BY created_at DESC LIMIT :n"), {"n": limit}).mappings().all()
    return {"analyses": [dict(r) for r in rows]}


@app.get("/v1/analyses/{analysis_id}", dependencies=[Depends(ops.require("read"))])
def get_analysis(analysis_id: str) -> Dict[str, Any]:
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM analyses WHERE id = :i"), {"i": analysis_id[:64]}).mappings().first()
    if r is None:
        raise HTTPException(404, "analysis not found")
    return {**{k: v for k, v in dict(r).items() if k != "findings"}, **json.loads(r["findings"])}
