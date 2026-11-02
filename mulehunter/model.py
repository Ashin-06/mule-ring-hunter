"""Directed GraphSAGE mule scorer, a gradient-boosting baseline and their ensemble.

Money flow has a direction: a mule *receives* from many unrelated people and *sends*
to other mules. The model therefore aggregates senders and receivers separately in
every layer instead of treating the graph as undirected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv
from xgboost import XGBClassifier

from .features import FEATURES, account_features, model_matrix

SCORED_KINDS = ("retail", "merchant")


class DirSAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.fwd = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(layers))
        self.rev = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden + d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout

    def embed(self, x, edge_index):
        h = F.relu(self.inp(x))
        rev = edge_index.flip(0)
        for f, r, n in zip(self.fwd, self.rev, self.norms):
            h = n(h + F.relu(f(h, edge_index) + r(h, rev)))  # residual keeps own behaviour visible
            h = F.dropout(h, self.dropout, self.training)
        return h

    def forward(self, x, edge_index):
        return self.head(torch.cat([self.embed(x, edge_index), x], dim=1)).squeeze(-1)


@dataclass
class Dataset:
    accounts: pd.DataFrame
    txns: pd.DataFrame
    feats: pd.DataFrame
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    eligible: np.ndarray
    split: np.ndarray  # "train" | "val" | "test" | ""


def build(accounts: pd.DataFrame, txns: pd.DataFrame, seed: int = 0, cutoff_ts: int | None = None) -> Dataset:
    """cutoff_ts: only use transactions before this moment (early-detection setting).
    Mules whose ring has not started yet are left out of training and scoring."""
    if cutoff_ts is not None:
        txns = txns[txns["ts"] < cutoff_ts]
    feats = account_features(accounts, txns)
    pairs = txns.groupby(["src", "dst"]).size().reset_index()[["src", "dst"]].to_numpy().T
    edge_index = torch.as_tensor(pairs, dtype=torch.long)
    eligible = accounts["kind"].isin(SCORED_KINDS).to_numpy()
    if cutoff_ts is not None and "ring_start" in accounts:
        # not &=: pandas 3 hands out read-only arrays
        eligible = eligible & ~((accounts["is_mule"] == 1) & (accounts["ring_start"] >= cutoff_ts)).to_numpy()
    split = ring_split(accounts, eligible, seed)
    split[~eligible] = ""
    return Dataset(accounts, txns, feats, torch.as_tensor(model_matrix(feats)), edge_index,
                   torch.as_tensor(accounts["is_mule"].to_numpy(), dtype=torch.float32), eligible, split)


def ring_split(accounts: pd.DataFrame, eligible: np.ndarray, seed: int = 0, frac=(0.5, 0.25)) -> np.ndarray:
    """Whole rings go to one split so the test set contains rings never seen in training."""
    rng = np.random.default_rng(seed)
    split = np.full(len(accounts), "", dtype=object)
    rings = accounts.loc[accounts["ring_id"] >= 0, "ring_id"].unique()
    rng.shuffle(rings)
    a, b = int(len(rings) * frac[0]), int(len(rings) * (frac[0] + frac[1]))
    ring_of = accounts["ring_id"].to_numpy()
    for name, rs in (("train", rings[:a]), ("val", rings[a:b]), ("test", rings[b:])):
        split[np.isin(ring_of, rs)] = name
    normal = np.flatnonzero(eligible & (ring_of < 0))
    # labelled data without ring ids (public datasets): plain random split
    if len(rings) == 0:
        normal = np.flatnonzero(eligible)
    rng.shuffle(normal)
    a, b = int(len(normal) * frac[0]), int(len(normal) * (frac[0] + frac[1]))
    split[normal[:a]], split[normal[a:b]], split[normal[b:]] = "train", "val", "test"
    return split


def metrics(y: np.ndarray, p: np.ndarray, budget: float = 0.01) -> dict:
    k = max(1, int(len(y) * budget))
    top = np.argsort(-p)[:k]
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        f"recall@top{budget:.0%}": float(y[top].sum() / max(y.sum(), 1)),
        f"precision@top{budget:.0%}": float(y[top].mean()),
    }


def train_gnn(ds: Dataset, epochs: int = 400, lr: float = 3e-3, seed: int = 0, device: str | None = None, verbose=False):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, ei, y = ds.x.to(device), ds.edge_index.to(device), ds.y.to(device)
    tr = torch.as_tensor(ds.split == "train", device=device)
    va = ds.split == "val"
    model = DirSAGE(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    pos_weight = (y[tr] == 0).sum() / (y[tr] == 1).sum().clamp(min=1)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, ei)[tr], y[tr], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            p = predict(model, ds, device)
            score = average_precision_score(ds.y.numpy()[va], p[va])
            if score > best:
                best, best_state, patience = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
            if verbose and epoch % 50 == 0:
                print(f"epoch {epoch:4d} loss {loss.item():.4f} val PR-AUC {score:.4f}")
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model.cpu()


@torch.no_grad()
def predict(model: DirSAGE, ds: Dataset, device: str = "cpu") -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(ds.x.to(device), ds.edge_index.to(device))).cpu().numpy()


def train_xgb(ds: Dataset, seed: int = 0) -> XGBClassifier:
    tr = ds.split == "train"
    y = ds.y.numpy()
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), random_state=seed,
                        eval_metric="aucpr", n_jobs=4)
    clf.fit(ds.feats.to_numpy()[tr], y[tr])
    return clf


def ensemble(p_gnn: np.ndarray, p_xgb: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Average of within-population ranks; robust to the two models' calibration."""
    out = np.zeros_like(p_gnn)
    for p in (p_gnn, p_xgb):
        r = np.zeros_like(p)
        r[eligible] = rankdata(p[eligible]) / eligible.sum()
        out += r / 2
    # squash ranks so high values read like probabilities
    return out ** 4


def evaluate(ds: Dataset, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    te = ds.split == "test"
    y = ds.y.numpy()[te]
    return pd.DataFrame({name: metrics(y, p[te]) for name, p in scores.items()}).T


def breakdown(ds: Dataset, p: np.ndarray, budget: float = 0.02) -> pd.DataFrame:
    """Recall per mule type when investigators review the top `budget` of test accounts."""
    te = np.flatnonzero((ds.split == "test") & ds.eligible)
    k = max(1, int(len(te) * budget))
    flagged = set(te[np.argsort(-p[te])[:k]])
    acc = ds.accounts.iloc[te]
    mules = acc[acc["is_mule"] == 1].copy()
    mules["caught"] = [i in flagged for i in mules.index]
    groups = {"role": mules["role"]}
    if "rented" in mules:
        groups["rented account"] = mules["rented"].map({True: "rented", False: "fresh"})
        groups["ring style"] = mules["ring_style"]
    rows = []
    for name, key in groups.items():
        for val, g in mules.groupby(key)["caught"]:
            rows.append({"group": name, "value": val, "mules": len(g), "recall": g.mean()})
    return pd.DataFrame(rows)


def fit_all(ds: Dataset, seed: int = 0, verbose: bool = False):
    gnn = train_gnn(ds, seed=seed, verbose=verbose)
    xgb = train_xgb(ds, seed)
    p_gnn = predict(gnn, ds)
    p_xgb = xgb.predict_proba(ds.feats.to_numpy())[:, 1]
    p_ens = ensemble(p_gnn, p_xgb, ds.eligible)
    for p in (p_gnn, p_xgb, p_ens):
        p[~ds.eligible] = 0.0
    return gnn, xgb, {"features + XGBoost": p_xgb, "directed GraphSAGE": p_gnn, "ensemble": p_ens}


def feature_saliency(model: DirSAGE, ds: Dataset, node: int) -> pd.Series:
    """Gradient x input on the node's own features: which behaviours drove its score."""
    model.eval()
    x = ds.x.clone().requires_grad_(True)
    model(x, ds.edge_index)[node].backward()
    sal = (x.grad[node] * x[node]).detach().numpy()
    return pd.Series(sal, index=FEATURES).sort_values(key=np.abs, ascending=False)
