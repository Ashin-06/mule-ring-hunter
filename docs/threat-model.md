# Threat model

What the analysis service protects, from whom, and what is still open. Written for the 1.0.0 code;
nobody independent has reviewed it.

## What it protects

| Asset | Where it lives |
|---|---|
| Submitted transactions (accounts, devices, amounts, times) | request bodies only; not stored |
| Findings (flagged accounts, scores, ring candidates) | `analyses` table |
| The models | `gnn.pt` (torch state dict) and `xgb.json` in `MULEHUNTER_MODEL_DIR` |
| API keys | callers hold the key; the service holds only SHA-256 hashes |

Findings are sensitive: they name customers suspected of laundering, and tipping them off is an
offence under PMLA.

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| Someone reads who is suspected | API keys with roles; no CORS unless listed | Every `viewer` sees every analysis |
| A malicious model file runs code | `gnn.pt` is loaded with `weights_only=True` (tensors only) and XGBoost from JSON | Anyone who can write the model directory can still swap in a weaker model |
| Mules evade detection | Graph and behavioural features; held-out-ring evaluation | Rented accounts are the weakest case (recall 0.86 simulated); real data is much harder (Elliptic F1 0.75, near 0.02 after a regime change) |
| A batch too small or unrepresentative skews scores | Documented: features are standardised within each batch | Not enforced beyond a minimum of 3 accounts |
| Flooding | 8 MB body cap, 20,000 accounts and 50,000 transactions per batch, per-client rate limit | Rate limit is per process; analysis is CPU-heavy |
| Errors leak data | Validation errors don't echo input; unhandled errors return only a request ID | — |

## Not covered

- Feeding the bank's full payment graph continuously (the API takes batches).
- Case management and reporting to FIU-IND.
- TLS (terminate it at the ingress).
