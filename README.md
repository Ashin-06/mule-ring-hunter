# Mule Ring Hunter

Finds money-mule *networks* in instant-payment data (UPI / IMPS style), not just single suspicious accounts. For each ring it rebuilds the laundering chain end to end: victims → entry mules → layering → cash-out.

Scam proceeds are laundered through chains of accounts that each look only mildly odd. Transaction-level fraud rules miss them because no single payment is unusual. The signal is in the shape: many unrelated people paying a new account that forwards nearly everything within minutes to accounts that drain it at ATMs.

## What it does

1. **Behavioural features per account.** Pass-through ratio, median holding time, share of money forwarded within an hour, one-off senders, structuring just below ₹50,000, cash/crypto exit share, device sharing, burstiness, account age.
2. **Directed GraphSAGE.** Each layer aggregates *senders* and *receivers* separately, so the model learns that a mule is paid by victims and pays other mules. A residual path keeps each account's own behaviour visible.
3. **XGBoost baseline and rank ensemble.** A gradient-boosted model on the same features, averaged with the GNN by rank, so calibration differences don't matter.
4. **Ring reconstruction.** Flagged accounts are linked by money flows, shared devices and shared victims. Louvain communities split hubs that would otherwise merge separate rings. For each ring the tool reports victims, money in, cash out, active window and each member's inferred role.
5. **Plain-language reasons**, for example: *"money out equals 97% of money in; payments leave a median 14 min after funds arrive · 23 different senders, 91% of whom paid only once · device shared with 3 other accounts"*.
6. **Early detection.** `timeline` retrains as of day N using only transactions up to that day, which shows how soon a ring becomes catchable.

## The data is the hard part

The simulator (`mulehunter/simulate.py`) builds a 6,000-customer payment network that includes the cases that cause false positives:

| Legitimate pattern | Why it looks like a mule |
|---|---|
| Merchants settling nightly | Fan-in from hundreds of customers, ~97% passed on |
| Group collectors (trip money, chit funds) | 10–20 friends chip in, the total is forwarded within an hour |
| Small traders / resellers | Many one-off buyers, supplier paid the same day |
| Landlords | Several fixed inflows, a large transfer out |
| Shared household phones | Device sharing |

The mule rings vary in the ways real rings do:
- **Rented accounts.** Half of the entry mules are real customers with years of salary and shopping history.
- **"Careful" rings** (40%) hold funds for hours, keep a 10–25% cut and don't share phones.
- Transfers are structured below reporting limits, and cash-out is split across ATM withdrawals or crypto exchanges.

Train/validation/test splits are **ring-aware**: every test ring is entirely unseen during training.

## Results

`python -m mulehunter train` (seed 7, 2% alert budget):

```
held-out rings (never seen in training):
                    pr_auc  roc_auc  recall@top1%  precision@top1%
features + XGBoost   0.977    0.999         0.536              1.0
directed GraphSAGE   0.995    1.000         0.536              1.0
ensemble             0.989    1.000         0.536              1.0

recall by mule type (test split):
         group   value  mules  recall
          role cashout      5    1.00
          role   tier1     14    0.93
          role   tier2      9    1.00
rented account   fresh     21    1.00
rented account  rented      7    0.86
```

Recall@top1% tops out at 0.54 here only because the test split has more mules than 1% of its accounts.

Early detection (`python -m mulehunter timeline`), PR-AUC on unseen rings using only the first N days:

| Day | Active test mules | XGBoost | GraphSAGE | Ensemble |
|---|---|---|---|---|
| 10 | 8 | 0.84 | 0.25 | 0.62 |
| 15 | 8 | 0.96 | 0.92 | 0.98 |
| 20 | 14 | 0.85 | 0.84 | 0.81 |
| 30 | 20 | 0.99 | 1.00 | 1.00 |
| 45 | 28 | 0.98 | 0.99 | 0.99 |

Ring reconstruction at a 2% budget recovers 13 of 16 true rings with Jaccard ≥ 0.5 (mean Jaccard 0.76).

These are synthetic numbers and the simulator is generous. Expect much lower scores on real data; use the loaders below to measure that. Rented accounts are where the model is weakest, and they are also where banks struggle most.

## Run it

```bash
pip install -r requirements.txt
python -m mulehunter simulate                 # data/accounts.csv.gz, data/txns.csv.gz
python -m mulehunter train                    # metrics, rings, artifacts/
python -m mulehunter timeline                 # early-detection curve
python -m mulehunter explain 6339 --saliency  # why was this account flagged
streamlit run app.py                          # investigation dashboard
python -m pytest -q                           # 6 tests
```

The dashboard has three views:
- **Model:** metrics, early-detection curve, score histogram.
- **Rings:** ring table, money-flow graph with inferred roles, Sankey of where the money went, per-member reasons.
- **Account lookup:** account history.

An alert-budget slider recomputes the flags and rings live.

## Real data

```bash
# IBM AMLworld (Altman et al., NeurIPS 2023) - accounts touching a laundering transaction are positives
python -m mulehunter train --ibm HI-Small_Trans.csv

# PaySim - destinations of fraudulent TRANSFERs are positives
python -m mulehunter train --paysim PS_20174392719_1491204439457_log.csv
```

For your own data, write two tables:
- `accounts`: `account, kind, created_ts, device, is_mule, ring_id`
- `txns`: `txn_id, ts, src, dst, amount, channel`

Put them in `data/` as `.csv.gz`. Device ids and account opening dates add a lot of signal if you have them.

## Layout

```
mulehunter/
  simulate.py   payment network + mule rings + loaders for IBM AML / PaySim
  features.py   behavioural features from the raw ledger
  model.py      directed GraphSAGE, XGBoost, ensemble, ring-aware split, metrics
  explain.py    reasons, ring reconstruction (flows + devices + shared victims, Louvain)
  __main__.py   CLI
app.py          Streamlit investigation dashboard
```

## Notes

- The model is transductive: it scores the graph it was trained on. For daily scoring, retrain on a rolling window, or keep the weights and call `predict` on the new graph. GraphSAGE generalises to unseen nodes.
- Mule labels in real banks are noisy and late. Accounts confirmed through chargebacks or police requests are positives. Treat unconfirmed accounts as unlabeled, not negative.
- Reasons are computed from features, not the model. They stay readable and auditable even when the model changes.
