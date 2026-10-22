import pandas as pd
import pytest

from mulehunter.explain import find_rings, reasons, ring_recovery
from mulehunter.features import account_features
from mulehunter.model import build, evaluate, fit_all
from mulehunter.simulate import DAY, SimConfig, simulate

SMALL = SimConfig(n_retail=1500, n_merchants=60, n_employers=15, n_collectors=15, n_landlords=20, n_traders=40, n_rings=8, days=30, seed=3)


@pytest.fixture(scope="module")
def world():
    return simulate(SMALL)


def test_simulator_builds_layered_rings(world):
    accounts, txns = world
    assert set(accounts.loc[accounts.is_mule == 1, "role"]) == {"tier1", "tier2", "cashout"}
    assert accounts.loc[accounts.is_mule == 1, "rented"].any()  # some mules are rented real accounts
    t1 = accounts.loc[accounts.role == "tier1", "account"]
    t2 = set(accounts.loc[accounts.role == "tier2", "account"])
    assert txns[txns.src.isin(t1) & txns.dst.isin(t2)].shape[0] > 0


def test_holding_time_and_passthrough():
    accounts = pd.DataFrame({"account": [0, 1, 2], "kind": "retail", "created_ts": 0, "device": ["a", "b", "c"]})
    txns = pd.DataFrame({
        "txn_id": range(4), "ts": [100, 700, 10_000, 100_000], "src": [0, 1, 0, 1], "dst": [1, 2, 1, 2],
        "amount": [1000.0, 980.0, 500.0, 400.0], "channel": "UPI",
    })
    f = account_features(accounts, txns)
    assert f.loc[1, "passthrough"] == pytest.approx(1380 / 1501, rel=1e-3)
    # first forward 600 s after receipt, second 90 000 s later -> median 45 300 s
    assert f.loc[1, "median_hold_min"] == pytest.approx((600 + 90_000) / 2 / 60)
    assert f.loc[1, "fast_forward_share"] == pytest.approx(980 / 1380)


def test_models_beat_chance_on_unseen_rings(world):
    accounts, txns = world
    ds = build(accounts, txns, seed=1)
    _, _, scores = fit_all(ds, seed=1)
    table = evaluate(ds, scores)
    base_rate = ds.y.numpy()[ds.split == "test"].mean()
    assert (table["pr_auc"] > 5 * base_rate).all()


def test_early_cutoff_hides_future_rings(world):
    accounts, txns = world
    ds = build(accounts, txns, cutoff_ts=8 * DAY)
    future = (accounts.is_mule == 1) & (accounts.ring_start >= 8 * DAY)
    assert not ds.eligible[future.to_numpy()].any()
    assert ds.txns.ts.max() < 8 * DAY


def test_rings_reconstructed_from_true_labels(world):
    accounts, txns = world
    perfect = accounts["is_mule"].to_numpy().astype(float)
    rings = find_rings(accounts, txns, perfect, 0.5)
    rec = ring_recovery(accounts, rings)
    assert rec["jaccard"].mean() > 0.8
    assert all(r.victim_inflow > 0 for r in rings)


def test_reasons_are_specific(world):
    accounts, txns = world
    f = account_features(accounts, txns)
    mule = int(accounts.loc[(accounts.role == "tier1") & ~accounts.rented, "account"].iloc[0])
    text = " ".join(reasons(f, mule))
    assert "money in" in text or "within an hour" in text
