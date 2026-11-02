# Model card: mule ring detection

An ensemble of a directed GraphSAGE network over the payment graph and XGBoost over per-account
behavioural features (holding time, pass-through, fan-in, structuring, cash-out share, shared
devices), averaged by rank. Flagged accounts are grouped into ring candidates. Numbers come from
`docs/evaluation.md` and `results/`.

## Intended use

Ranking accounts for a bank's financial-intelligence or fraud team to review as possible money mules,
within an alert budget. Not for freezing accounts or reporting without human review.

## Evaluation

**Simulated payment network** (6,000 customers, 16 rings, legitimate look-alikes such as merchants,
group collectors and landlords; test rings never seen in training): PR-AUC 0.976 (XGBoost), 0.998
(GraphSAGE), 0.988 (ensemble); 15 of 16 rings rebuilt at a 2% alert budget. Rented accounts are the
weakest case (recall 0.86).

**Real Bitcoin flows** (Elliptic, time split): illicit F1 0.75, close to the published random forest
(0.79). After a dark market closed (step 43 on) every model drops to F1 about 0.02.

## Known failure modes

- The simulator is generous: expect much lower scores on real bank data.
- Regime changes (a market closing, a new laundering method) break every model, as Elliptic shows.
- Features are standardised within each analysed batch, so very small or unusual batches skew scores.
- Rented accounts with long normal histories are the hardest to catch.

## Before relying on it

Train and evaluate on the bank's own confirmed mule cases with a ring-level split, and monitor for
drift after every change in fraud patterns.
