# Evaluation

## Simulated payment network

`mulehunter/simulate.py` builds a 6,000-customer payment network. It deliberately includes legitimate patterns that look like mule activity:

| Legitimate pattern | Why it looks like a mule |
|---|---|
| Merchants settling nightly | money in from hundreds of customers, about 97% passed on |
| Group collectors (trip money, chit funds) | 10–20 friends pay in, the total is forwarded within an hour |
| Small traders and resellers | many one-off buyers, supplier paid the same day |
| Landlords | several fixed payments in, one large transfer out |
| Shared household phones | several accounts on one device |

The mule rings vary in realistic ways. Half of the entry mules are rented accounts with years of normal history. 40% of rings are "careful": they hold money for hours, keep a 10–25% cut and don't share phones. Transfers are kept below reporting limits and cash-out is split across ATMs or crypto exchanges.

Train, validation and test splits are made by ring, so every test ring is unseen during training.

`python -m mulehunter train` (seed 7, 2% alert budget), on held-out rings. These numbers are from a re-run on the pinned dependencies (torch 2.15, PyG 2.8, xgboost 3.4); the first run gave PR-AUC 0.977 / 0.995 / 0.989 and rebuilt 13 of 16 rings. The early-detection table below and the Elliptic results were not re-run.

| Model | PR-AUC | ROC-AUC | Recall in top 1% | Precision in top 1% |
|---|---|---|---|---|
| Features + XGBoost | 0.976 | 0.999 | 0.536 | 1.0 |
| Directed GraphSAGE | 0.998 | 1.000 | 0.536 | 1.0 |
| Ensemble | 0.988 | 1.000 | 0.536 | 1.0 |

Recall in the top 1% is capped at 0.54 because the test split has more mules than 1% of its accounts.

Recall by mule type: cash-out 1.00, first-tier 0.93, second-tier 1.00, freshly opened accounts 1.00, rented accounts 0.86.

Early detection (`python -m mulehunter timeline`), PR-AUC on unseen rings using only the first N days:

| Day | Active test mules | XGBoost | GraphSAGE | Ensemble |
|---|---|---|---|---|
| 10 | 8 | 0.84 | 0.25 | 0.62 |
| 15 | 8 | 0.96 | 0.92 | 0.98 |
| 20 | 14 | 0.85 | 0.84 | 0.81 |
| 30 | 20 | 0.99 | 1.00 | 1.00 |
| 45 | 28 | 0.98 | 0.99 | 0.99 |

Ring reconstruction at a 2% budget recovers 15 of 16 true rings with Jaccard ≥ 0.5 (mean 0.79).

The simulator is generous, so expect much lower scores on real data. Rented accounts are the weakest case, as they are for banks.

## Real illicit flows: Elliptic

No bank publishes labelled mule accounts, so the closest real test is the [Elliptic Bitcoin dataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.EllipticBitcoinDataset.html): 203,769 real transactions, 234,355 payment flows and 4,545 transactions labelled illicit. Its nodes are transactions rather than accounts, so the same models were run on Elliptic's own 165 features, with the paper's time split: train on steps 1–34, test on 35–49.

`python -m mulehunter elliptic --data elliptic/` writes `results/elliptic.json`.

| Model (test steps 35–49, 1,083 illicit) | PR-AUC | Illicit F1 |
|---|---|---|
| Features + XGBoost, paper protocol | 0.80 | 0.75 |
| Features + XGBoost, threshold tuned on steps 30–34 | 0.79 | 0.72 |
| Directed GraphSAGE | 0.50–0.55 | 0.48–0.52 |
| Ensemble | 0.75 | 0.73 |
| Published (Weber et al. 2019): random forest / GCN | | 0.79 / 0.63 |

The feature model is close to the published random forest, and the graph model trails it, as the published GCN does. Every model collapses after step 43 (F1 about 0.02), when a dark market closed and illicit behaviour changed; the paper reports the same. Models like this need retraining on a rolling window and drift monitoring.
