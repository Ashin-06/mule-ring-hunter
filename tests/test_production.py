"""The analysis API: settings, safe model loading, auth, limits, headers, metrics, health, the
analysis log and migrations."""
import hashlib
import importlib
import json
import os
import subprocess
import sys

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import text

from mulehunter import db, ops
from mulehunter.model import build, train_gnn, train_xgb
from mulehunter.simulate import SimConfig, simulate

SMALL = SimConfig(n_retail=800, n_merchants=40, n_employers=10, n_collectors=10, n_landlords=10, n_traders=20, n_rings=5, days=20, seed=5)
KEY = "mulehunter_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "fiu", "role": "analyst"}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    accounts, txns = simulate(SMALL)
    ds = build(accounts, txns, seed=0)
    d = tmp_path_factory.mktemp("models")
    torch.save(train_gnn(ds, epochs=40, seed=0).state_dict(), d / "gnn.pt")
    train_xgb(ds, seed=0).save_model(d / "xgb.json")
    body = {"accounts": [{"account": f"A{r.account}", "kind": r.kind, "created_ts": int(r.created_ts), "device": str(r.device or "")}
                         for r in accounts.itertuples()],
            "txns": [{"ts": int(r.ts), "src": f"A{r.src}", "dst": f"A{r.dst}", "amount": float(r.amount), "channel": r.channel}
                     for r in txns.head(20_000).itertuples()]}
    return d, body


def boot(monkeypatch, models, **env):
    for k in [k for k in os.environ if k.startswith("MULEHUNTER_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("MULEHUNTER_MODEL_DIR", str(models))
    for k, v in env.items():
        monkeypatch.setenv(f"MULEHUNTER_{k}", v)
    import mulehunter.api as api

    return importlib.reload(api)


def test_production_settings_and_models(monkeypatch, world, tmp_path):
    models, _ = world
    env = {"MULEHUNTER_ENV": "production", "MULEHUNTER_API_KEYS": json.dumps(HASHED), "MULEHUNTER_DATABASE_URL": "sqlite://"}
    assert ops.load_settings(env).production
    for change, message in (({"MULEHUNTER_API_KEYS": None}, "API keys"), ({"MULEHUNTER_DATABASE_URL": None}, "DATABASE_URL"),
                            ({"MULEHUNTER_API_KEYS": json.dumps({KEY: {"user": "f", "role": "analyst"}})}, "sha256")):
        with pytest.raises(ops.ConfigError, match=message):
            ops.load_settings({k: v for k, v in {**env, **change}.items() if v is not None})
    with pytest.raises(ops.ConfigError, match="cannot load models"):
        boot(monkeypatch, tmp_path)  # an empty directory
    (tmp_path / "gnn.pt").write_bytes(b"not a state dict")
    with pytest.raises(ops.ConfigError, match="cannot load models"):
        boot(monkeypatch, tmp_path)
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"MULEHUNTER_HOST": "0.0.0.0"})


def test_analyses_find_mules_and_are_kept_across_a_restart(monkeypatch, world, tmp_path):
    models, body = world
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'a.db'}"
    c = TestClient(boot(monkeypatch, models, DATABASE_URL=url).app)
    r = c.post("/v1/analyse", json={**body, "alert_budget": 0.05, "label": "region-7 week 3"}).json()
    assert r["flagged"] and all(f["account"].startswith("A") for f in r["flagged"])
    assert r["flagged"] == sorted(r["flagged"], key=lambda f: -f["score"])
    bad = {**body, "txns": body["txns"][:5] + [{"ts": 1, "src": "A0", "dst": "NOBODY", "amount": 10.0}]}
    assert c.post("/v1/analyse", json=bad).status_code == 422
    dup = {**body, "accounts": body["accounts"] + body["accounts"][:1]}
    assert c.post("/v1/analyse", json=dup).status_code == 422

    c = TestClient(boot(monkeypatch, models, DATABASE_URL=url).app)  # restart
    got = c.get(f"/v1/analyses/{r['analysis_id']}").json()
    assert got["label"] == "region-7 week 3" and got["flagged"] == r["flagged"] and got["rings"] == r["rings"]
    assert any(a["id"] == r["analysis_id"] for a in c.get("/v1/analyses").json()["analyses"])
    assert c.get("/v1/analyses/nope").status_code == 404


def test_roles_metrics_limits_and_errors(monkeypatch, world):
    models, body = world
    keys = {"k-an": {"user": "fiu", "role": "analyst"}, "k-view": {"user": "vic", "role": "viewer"},
            "k-mon": {"user": "prom", "role": "monitor"}}
    c = TestClient(boot(monkeypatch, models, API_KEYS=json.dumps(keys)).app)
    small = {**body, "txns": body["txns"][:2000]}
    assert c.post("/v1/analyse", json=small).status_code == 401
    assert c.post("/v1/analyse", json=small, headers={"x-api-key": "k-view"}).status_code == 403
    assert c.post("/v1/analyse", json=small, headers={"x-api-key": "k-an"}).status_code == 200
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'mulehunter_decisions_total{decision="accounts_flagged"}' in m.text

    api = boot(monkeypatch, models, MAX_BODY_BYTES="5000", RATE_LIMIT="3")

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: account 1234567890")

    c = TestClient(api.app)
    assert c.post("/v1/analyse", json=small).status_code == 413
    crash = c.get("/boom")
    assert crash.status_code == 500 and "1234567890" not in crash.text and crash.json()["request_id"]
    bad = c.post("/v1/analyse", json={"accounts": [{"account": "SECRET", "kind": "bank"}], "txns": []})
    assert bad.status_code == 422 and "SECRET" not in bad.text
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/v1/analyses").status_code == 429  # four requests counted, limit 3


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"] and db.migrate(eng) == []
    with eng.connect() as c:
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert {"analyses", "schema_migrations"} <= tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("MULEHUNTER_")}, "MULEHUNTER_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "mulehunter", "migrate"], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0 and "up to date" in out.stdout
