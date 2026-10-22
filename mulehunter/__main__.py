"""Command line.

    python -m mulehunter simulate --out data
    python -m mulehunter train --data data --out artifacts
    python -m mulehunter train --ibm HI-Small_Trans.csv --out artifacts
    python -m mulehunter timeline --data data
    python -m mulehunter explain 1234 --out artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .explain import find_rings, reasons, ring_recovery
from .model import breakdown, build, evaluate, feature_saliency, fit_all
from .simulate import DAY, SimConfig, load_ibm_aml, load_paysim, simulate


def load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if getattr(args, "ibm", None):
        return load_ibm_aml(args.ibm)
    if getattr(args, "paysim", None):
        return load_paysim(args.paysim)
    d = Path(args.data)
    if not (d / "accounts.csv.gz").exists():
        print(f"no data in {d}, simulating")
        a, t = simulate(SimConfig(seed=args.seed))
        d.mkdir(parents=True, exist_ok=True)
        a.to_csv(d / "accounts.csv.gz", index=False)
        t.to_csv(d / "txns.csv.gz", index=False)
        return a, t
    a = pd.read_csv(d / "accounts.csv.gz", keep_default_na=False)
    if "rented" in a:
        a["rented"] = a["rented"].astype(str) == "True"
    return a, pd.read_csv(d / "txns.csv.gz")


def alert_threshold(scores: np.ndarray, eligible: np.ndarray, budget: float) -> float:
    return float(np.quantile(scores[eligible], 1 - budget))


def cmd_train(args):
    accounts, txns = load(args)
    cutoff = args.cutoff_day * DAY if args.cutoff_day else None
    ds = build(accounts, txns, args.seed, cutoff)
    gnn, xgb, scores = fit_all(ds, args.seed, verbose=True)
    table = evaluate(ds, scores)
    print("\nheld-out rings (never seen in training):")
    print(table.round(3).to_string())
    p = scores["ensemble"]
    thr = alert_threshold(p, ds.eligible, args.budget)
    rings = find_rings(accounts, ds.txns, p, thr)
    rec = ring_recovery(accounts, rings)
    print(f"\nalert budget {args.budget:.0%} -> threshold {thr:.3f}, {len(rings)} ring candidates")
    if len(rec):
        print(f"true rings recovered (Jaccard >= 0.5): {(rec['jaccard'] >= 0.5).mean():.0%}, mean Jaccard {rec['jaccard'].mean():.2f}")
    if "rented" in accounts:
        print("\nrecall by mule type (test split):")
        print(breakdown(ds, p, args.budget).round(2).to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account": accounts["account"], **{k: v for k, v in scores.items()}}).rename(
        columns={"ensemble": "score"}).to_csv(out / "scores.csv", index=False)
    ds.feats.to_csv(out / "features.csv")
    torch.save(gnn.state_dict(), out / "gnn.pt")
    xgb.save_model(out / "xgb.json")
    table.to_json(out / "metrics.json", indent=2)
    (out / "rings.json").write_text(
        json.dumps([{k: v for k, v in r.__dict__.items() if k != "flows"} for r in rings], indent=2, default=int),
        encoding="utf-8",
    )
    (out / "run.json").write_text(
        json.dumps({"threshold": thr, "budget": args.budget, "cutoff_day": args.cutoff_day}),
        encoding="utf-8",
    )
    print(f"\nartifacts written to {out}/")


def cmd_timeline(args):
    accounts, txns = load(args)
    rows = []
    for day in args.days:
        ds = build(accounts, txns, args.seed, day * DAY)
        _, _, scores = fit_all(ds, args.seed)
        t = evaluate(ds, scores)
        rows.append({"day": day, "active_test_mules": int(ds.y.numpy()[ds.split == "test"].sum()),
                     **{f"{m} PR-AUC": round(float(t.loc[m, "pr_auc"]), 3) for m in t.index}})
        print(rows[-1])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.out) / "timeline.csv", index=False)


def cmd_explain(args):
    out = Path(args.out)
    scores = pd.read_csv(out / "scores.csv").set_index("account")
    feats = pd.read_csv(out / "features.csv").set_index("account")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    accounts, txns = load(args)
    flagged = set(scores.index[scores["score"] >= run["threshold"]])
    s = scores.loc[args.account]
    print(f"account {args.account}: score {s['score']:.3f} (GNN {s['directed GraphSAGE']:.3f}, XGB {s['features + XGBoost']:.3f})"
          f" -> {'FLAGGED' if args.account in flagged else 'not flagged'}")
    for r in reasons(feats, args.account, flagged, txns):
        print(" -", r)
    if args.saliency:
        from .model import DirSAGE

        ds = build(accounts, txns, args.seed, run["cutoff_day"] * DAY if run["cutoff_day"] else None)
        m = DirSAGE(ds.x.shape[1])
        m.load_state_dict(torch.load(out / "gnn.pt", weights_only=True))
        print("\nmodel attribution (gradient x input):")
        print(feature_saliency(m, ds, args.account).head(6).round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mulehunter")
    ap.add_argument("--seed", type=int, default=7)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--out", default="data")
    s.add_argument("--accounts", type=int, default=6000)
    s.add_argument("--rings", type=int, default=16)
    s.add_argument("--days", type=int, default=45)
    for name in ("train", "timeline", "explain"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data")
        p.add_argument("--ibm", help="IBM AMLworld transactions CSV")
        p.add_argument("--paysim", help="PaySim CSV")
        p.add_argument("--out", default="artifacts")
        if name == "train":
            p.add_argument("--cutoff-day", type=int, help="only use the first N days (early detection)")
            p.add_argument("--budget", type=float, default=0.02, help="share of accounts investigators can review")
        if name == "timeline":
            p.add_argument("--days", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40])
        if name == "explain":
            p.add_argument("account", type=int)
            p.add_argument("--saliency", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        a, t = simulate(SimConfig(n_retail=args.accounts, n_rings=args.rings, days=args.days, seed=args.seed))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        a.to_csv(Path(args.out) / "accounts.csv.gz", index=False)
        t.to_csv(Path(args.out) / "txns.csv.gz", index=False)
        print(f"{len(a)} accounts ({a['is_mule'].sum()} mules in {a.loc[a.ring_id >= 0, 'ring_id'].nunique()} rings), {len(t)} transactions -> {args.out}/")
    else:
        {"train": cmd_train, "timeline": cmd_timeline, "explain": cmd_explain}[args.cmd](args)


if __name__ == "__main__":
    main()
