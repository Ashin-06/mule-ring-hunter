# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-11-02

### Added
- An analysis API (`python -m mulehunter serve`): `POST /v1/analyse` scores a batch of accounts and
  transactions with the trained ensemble, flags accounts within an alert budget and returns ring
  candidates; `GET /v1/analyses[/{id}]`. Findings are logged in SQLite or Postgres with numbered
  migrations (`python -m mulehunter migrate`); submitted transactions are not stored.
- Models load from `gnn.pt` with `weights_only=True` and from `xgb.json`, so loading never runs code.
- Settings from `MULEHUNTER_*` environment variables, checked at startup; production refuses to start
  without hashed API keys, a database and a model directory. Roles on keys, rate limit, body cap,
  security headers, JSON logs with request IDs, `/healthz`, `/readyz` and Prometheus `/metrics`.
- `pyproject.toml`, pinned lock files (CPU-only torch from PyTorch's index, `xgboost-cpu`), a container
  image (slim Python 3.12, non-root, health check), CI with ruff, bandit, pip-audit and coverage.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`, MIT licence.

### Fixed
- With pandas 3 (the locked version), `build` failed on an early-detection cutoff because it modified
  a read-only array in place.

### Changed
- The simulated benchmark was re-run on the pinned dependencies: PR-AUC 0.976 / 0.998 / 0.988 (was
  0.977 / 0.995 / 0.989) and 15 of 16 rings rebuilt (was 13). The early-detection table and the
  Elliptic results were not re-run.
