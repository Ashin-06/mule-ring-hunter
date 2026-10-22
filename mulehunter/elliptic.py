"""Run the same directed GraphSAGE + XGBoost ensemble on real illicit-flow data.

Elliptic Bitcoin dataset (Weber et al., KDD 2019 workshop): 203,769 real Bitcoin
transactions, 234,355 directed payment flows between them, 166 anonymised features, and
labels for 4,545 illicit and 42,019 licit transactions (the rest unknown).
Files: https://data.pyg.org/datasets/elliptic/  (elliptic_txs_{features,edgelist,classes}.csv)

Nodes here are transactions rather than bank accounts, so the account features in
features.py (holding time, pass-through, device sharing) do not apply; the graph model
and the ensemble are used unchanged on the provided features. Split as in the paper:
train on time steps 1-34 (30-34 held out for early stopping and thresholds), test on
35-49, which includes the dark-market shutdown after step 43.

    python -m mulehunter elliptic --data path/to/elliptic/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .model import Dataset, ensemble, predict, train_gnn, train_xgb


def load(root: str | Path) -> tuple[Dataset, np.ndarray]:
    root = Path(root)
    feats = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    ids = feats.pop(0).to_numpy()
    step = feats.pop(1).to_numpy()  # time step: used for the split, not as a feature
    feats.columns = [f"f{i}" for i in range(feats.shape[1])]
    index = pd.Series(np.arange(len(ids)), index=ids)

    cls = pd.read_csv(root / "elliptic_txs_classes.csv").set_index("txId")["class"].reindex(ids).astype(str).to_numpy()
    y = (cls == "1").astype(np.float32)  # "1" illicit, "2" licit, "unknown"
    labelled = cls != "unknown"

    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    edge_index = torch.as_tensor(np.vstack([index[edges["txId1"]].to_numpy(), index[edges["txId2"]].to_numpy()]), dtype=torch.long)

    split = np.full(len(ids), "", dtype=object)
    split[labelled & (step <= 29)] = "train"
    split[labelled & (step >= 30) & (step <= 34)] = "val"
    split[labelled & (step >= 35)] = "test"
    x = feats.to_numpy(dtype=np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    ds = Dataset(pd.DataFrame(), pd.DataFrame(), feats, torch.as_tensor(x), edge_index, torch.as_tensor(y), labelled, split)
    return ds, step


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def run(root: str | Path, seed: int = 0) -> dict:
    ds, step = load(root)
    y = ds.y.numpy()
    gnn = train_gnn(ds, seed=seed)
    p_gnn = predict(gnn, ds)
    p_xgb = train_xgb(ds, seed).predict_proba(ds.feats.to_numpy())[:, 1]
    p_ens = ensemble(p_gnn, p_xgb, ds.eligible)

    va, te = ds.split == "val", ds.split == "test"
    out = {"dataset": "Elliptic Bitcoin (real transactions)", "nodes": int(len(y)), "edges": int(ds.edge_index.shape[1]),
           "labelled_illicit": int(y[ds.eligible].sum()), "test_illicit": int(y[te].sum()), "test_licit": int((y[te] == 0).sum()),
           "published_illicit_f1": {"random forest, all features (Weber et al.)": 0.788, "GCN (Weber et al.)": 0.628},
           "models": {}}
    for name, p in (("features + XGBoost", p_xgb), ("directed GraphSAGE", p_gnn), ("ensemble", p_ens)):
        thr = _best_threshold(y[va], p[va])
        row = {"pr_auc": round(float(average_precision_score(y[te], p[te])), 4),
               "roc_auc": round(float(roc_auc_score(y[te], p[te])), 4),
               "illicit_f1_val_threshold": round(float(f1_score(y[te], p[te] >= thr)), 4)}
        if name != "ensemble":  # the ensemble outputs ranks, so 0.5 means nothing for it
            row["illicit_f1_at_0.5"] = round(float(f1_score(y[te], p[te] >= 0.5)), 4)
        before, after = te & (step <= 42), te & (step >= 43)
        row["illicit_f1_steps_35_42"] = round(float(f1_score(y[before], p[before] >= thr)), 4)
        row["illicit_f1_steps_43_49"] = round(float(f1_score(y[after], p[after] >= thr)), 4)
        out["models"][name] = row

    # like for like with the paper: features only, trained on all of steps 1-34, threshold 0.5
    paper = Dataset(ds.accounts, ds.txns, ds.feats, ds.x, ds.edge_index, ds.y, ds.eligible,
                    np.where(ds.split == "val", "train", ds.split))
    p = train_xgb(paper, seed).predict_proba(ds.feats.to_numpy())[:, 1]
    out["models"]["features + XGBoost, paper protocol"] = {
        "pr_auc": round(float(average_precision_score(y[te], p[te])), 4),
        "illicit_f1_at_0.5": round(float(f1_score(y[te], p[te] >= 0.5)), 4),
    }
    return out
