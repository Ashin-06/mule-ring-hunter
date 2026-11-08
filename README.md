# Mule Ring Hunter

Finds money-mule networks in instant-payment data (UPI and IMPS style), not just single suspicious accounts. For each ring it traces the laundering chain from victims through entry mules and layering accounts to cash-out.

Scam money is moved through chains of accounts that each look only slightly unusual, so rules that judge one payment at a time miss them. The pattern shows in the network: many unrelated people paying a new account that forwards almost everything within minutes to accounts that empty it at ATMs.

## Features

- Behavioural features for every account: pass-through ratio, how long money stays, share forwarded within an hour, one-off senders, amounts just under ₹50,000, cash and crypto exits, shared devices and account age.
- A directed GraphSAGE model that looks at who pays an account and whom it pays separately.
- An XGBoost model on the same features, combined with the graph model by rank.
- Ring reconstruction: flagged accounts are linked by money flows, shared devices and shared victims, then split into rings with Louvain community detection. Each ring gets victims, money in, cash out, active dates and a role for every member.
- Plain-language reasons for every flag, for example: "money out equals 97% of money in; payments leave a median 14 min after funds arrive; 23 different senders, 91% of whom paid only once".
- An early-detection view that retrains on the first N days of data to show how soon a ring can be caught.
- A Streamlit dashboard for investigators.

## Quick start

```bash
pip install -r requirements.txt   # on macOS, first: brew install libomp
python -m pytest -q
python -m mulehunter simulate                 # writes data/accounts.csv.gz and data/txns.csv.gz
python -m mulehunter train                    # trains the models, prints metrics and rings
python -m mulehunter timeline                 # early-detection curve
python -m mulehunter explain 6339 --saliency  # why one account was flagged
streamlit run app.py                          # dashboard
```

The dashboard has three views: model metrics, the rings (with a money-flow graph and where the money went), and an account lookup. An alert-budget slider recomputes the flags and rings.

## Results

| Test | Result |
|---|---|
| Simulated network, rings never seen in training | PR-AUC 0.976–0.998; 15 of 16 rings rebuilt |
| Real Bitcoin flows (Elliptic, time split as in the paper) | illicit F1 0.75, close to the published random forest (0.79) |
| Elliptic after a dark market closed (step 43 on) | every model drops to F1 about 0.02 |

The simulated numbers measure the method on generated rings, not real-world accuracy. Details are in [docs/evaluation.md](docs/evaluation.md).

## Analysis API

`python -m mulehunter serve` (or the container image, `docker build -t mule-ring-hunter .`, see [docs/runbook.md](docs/runbook.md)) analyses a batch of accounts and transactions with the trained models.

| Endpoint | Purpose |
|---|---|
| `POST /v1/analyse` | up to 20,000 accounts and 50,000 transactions; returns scored and flagged accounts within an alert budget, and ring candidates |
| `GET /v1/analyses`, `/v1/analyses/{id}` | past analyses and their findings (the transactions are not stored) |
| `GET /healthz`, `/readyz`, `/metrics` | health, readiness and Prometheus metrics |

Models load from `gnn.pt` (with `weights_only=True`) and `xgb.json`, neither of which runs code. Features are standardised within each batch, so send a meaningful slice of the payment graph. API keys carry a role: `analyst`, `viewer`, `monitor`. Create one with `python -m mulehunter.ops keygen`.

| Variable | Purpose |
|---|---|
| `MULEHUNTER_ENV=production` | refuse to start without hashed API keys, a database and a model directory |
| `MULEHUNTER_MODEL_DIR` | `gnn.pt` and `xgb.json` from `train` |
| `MULEHUNTER_API_KEYS` / `MULEHUNTER_KEYS_FILE` | hashed API keys with roles |
| `MULEHUNTER_DATABASE_URL` | the analysis log: `sqlite:///...` or `postgresql://...` |

## Using your own data

Two public datasets can be loaded directly:

```bash
python -m mulehunter train --ibm HI-Small_Trans.csv          # IBM AMLworld
python -m mulehunter train --paysim PS_2017_log.csv          # PaySim
python -m mulehunter elliptic --data elliptic/               # Elliptic Bitcoin
```

For your own data, write two tables to `data/` as `.csv.gz`:

- `accounts`: `account, kind, created_ts, device, is_mule, ring_id`
- `txns`: `txn_id, ts, src, dst, amount, channel`

Device IDs and account opening dates add a lot of signal if you have them.

## Project layout

```
mulehunter/
  simulate.py   simulated payment network and loaders for IBM AML and PaySim
  features.py   behavioural features from the transactions
  model.py      GraphSAGE, XGBoost, ensemble, ring-aware splits, metrics
  explain.py    reasons and ring reconstruction
  elliptic.py   the same models on the Elliptic dataset
app.py          Streamlit dashboard
```

## Limitations

- The model scores the graph it was trained on. For daily scoring, retrain on a rolling window or call `predict` on the new graph.
- Real mule labels are noisy and late. Treat accounts confirmed through chargebacks or police requests as positives and the rest as unlabelled, not as negatives.
- Reasons come from the features, not the model, so they stay readable when the model changes.
- The code has not had an independent security review.

See [SECURITY.md](SECURITY.md), [docs/threat-model.md](docs/threat-model.md) and [docs/model-card.md](docs/model-card.md).

## License

MIT. See [LICENSE](LICENSE).
