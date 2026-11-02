# Runbook

How to train, deploy and operate the analysis service. Commands assume the repository root.

## Train

```bash
python -m mulehunter train --data data --out artifacts     # or --ibm / --paysim for those datasets
```

This writes `gnn.pt` and `xgb.json` (plus scores and metrics) to `artifacts/`. Train on the bank's own
labelled mule cases before real use.

## Deploy

```bash
docker build -t mule-ring-hunter:1.0.0 .
python -m mulehunter.ops keygen            # API keys per analyst, viewer and monitor
docker run -d --name mulehunter -p 8002:8002 -v mulehunter-data:/data -v $PWD/artifacts:/models:ro \
  -e MULEHUNTER_KEYS_FILE=/run/secrets/mulehunter-keys.json \
  mule-ring-hunter:1.0.0
curl -fsS http://127.0.0.1:8002/readyz
```

The image runs in production mode as a non-root user. It refuses to start, and logs why, without
keys, a database URL or readable models in `/models`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `MULEHUNTER_ENV` | — | `development` (default) or `production` |
| `MULEHUNTER_API_KEYS` / `MULEHUNTER_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `analyst` (analyse, read), `viewer`, `monitor`, `admin` |
| `MULEHUNTER_MODEL_DIR` | yes | `gnn.pt` and `xgb.json` from `train` (development default: `artifacts/`) |
| `MULEHUNTER_DATABASE_URL` | yes | the analysis log: `sqlite:////data/mulehunter.db` or `postgresql://...` |
| `MULEHUNTER_HOST`, `MULEHUNTER_PORT` | — | default `127.0.0.1:8002` |
| `MULEHUNTER_RATE_LIMIT`, `MULEHUNTER_MAX_BODY_BYTES` | — | default 120/min per client, 8,000,000 bytes |

## Watch

- `GET /healthz`; `GET /readyz` (database and models); `GET /metrics` (role `monitor`): requests,
  latency, errors, `mulehunter_decisions_total{decision="accounts_flagged|rings_found"}`.

## Rotate keys

Add the new API key hash, restart, hand over the key, remove the old hash, restart.

## Back up and restore

Back up the analysis log like any SQLite or Postgres database; keep model files with your model
artefacts. Findings name suspected customers: restrict access to backups.

## Incidents

1. **Findings disclosed outside the investigation team.** Treat as a possible tipping-off breach under
   PMLA; follow the bank's procedure.
2. **Leaked API key.** Remove its hash and restart; review that user's analyses.
