# Incident report: NYC taxi demand collapse, March 2020

*A replay, not a live outage.* `make simulate` runs the production loop one day at a time over March and April 2020: score the day, check drift, decide. All numbers below come from `reports/incident_summary.json` and `reports/incident_timeline.csv` and are reproduced by `make all`.

![incident timeline](reports/incident_timeline.png)

## Summary

* **Model:** next-hour yellow-taxi pickups per zone, for the 30 busiest zones. Version 1 was trained on 1 Jan to 14 Feb 2020 and validated on 15 to 29 Feb, with WAPE 11.2%.
* **What happened:** mean demand per zone-hour fell from 185 pickups (1 Mar) to 30 (19 Mar) and then 8 (end of March). Demand stayed at about 7 to 8 through April.
* **First signal:** layer 2, input drift. The PSI of the lag features crossed 0.25 on **8 March**, eleven days before the worst day.
* **Result over the 61 days:**

| | mean MAE | mean WAPE |
|---|---|---|
| no reaction (v1 frozen) | 23.41 | 171% |
| our system (gated retraining, human approval) | 12.55 | 38% |

## Timeline

| Date | Demand | What the system saw | Decision |
|---|---|---|---|
| 1 to 7 Mar | 185 | PSI 0.04 to 0.18, "watch" twice | nothing |
| **8 Mar** | 168 | **first PSI alarm (0.28)**, WAPE 20% vs 11% at deploy: it hurts | retrain, **refused by the gate** (candidate 3% worse) |
| 9 to 12 Mar | | PSI drops back, then 0.31 once | "drift but no harm": keep watching |
| 13 to 18 Mar | 139 to 45 | PSI 0.7 then 2.9 to 5.3, every lag | 6 more retrainings, **all refused** (candidates 16% to 58% worse) |
| **19 Mar** | 30 | WAPE 150% | candidate +57.8% better: **v2 approved** |
| 21 Mar | 10 | v2 already outdated | **v3** (+56%) |
| 22 to 23 Mar | | | refused (+4.7% and -3%, below the 5% bar) |
| 24 Mar | | | **v4** (+11.8%) |
| 25 to 28 Mar | | drift, no harm | keep watching |
| 29 Mar | 9 | | **v5** (+17.7%) |
| 5 Apr | 7 | | **v6** (+14.3%), last promotion |
| 12, 19, 26 Apr | 7 to 8 | weekly hurt checks | refused (+0.9%, +2.5%, +3%): not worth a release |

Total: 17 retrainings started, 5 promoted, 12 refused.

## Which layer noticed first

1. **Layer 1, infrastructure:** never fired. The API was healthy the whole time and answered in a few milliseconds. A purely technical dashboard would have shown green.
2. **Layer 2, input drift (PSI on lag_1, lag_24, lag_168):** first, on 8 March, on `lag_1` (the last hour reacts before yesterday or last week). `lag_24` joined on 14 March and `lag_168` on 15 March.
3. **Layer 3, prediction drift:** the same day as layer 2 (8 March). The predictions are mostly a function of `lag_1`, so they move with it. Layers 2 and 3 tell you something changed, not that it matters.
4. **Layer 4, outcome (WAPE against the true counts, one hour later):** this one decides. It confirmed the harm (20% on 8 March against 11% at deploy, then 150% on 19 March). It also stopped retraining when the drift did not hurt (9 March, 20 and 25 to 28 March).

## Why the gate refused seven times, and why that was right

From 8 to 18 March the regime was still moving. A candidate trained on the 14 days before the gate window had learned mostly normal February and early March traffic. It was always one step behind and scored worse than v1 on the 3 held-out days. Promoting it would have shipped a model that was worse *and* new. The gate held until 19 March, when the training window finally contained enough of the collapse. Then it let through a candidate 58% better.

The 5% bar also stopped churn in April. Candidates were 1% to 3% better, which is noise and not worth a release, a cache flush and a new thing to roll back.

## Rollback

Measured on the running API: `make rollback` moved `@production` from v6 to v5 and the API reloaded in **1.13 s**. Rolling all the way back to v1 took four commands, about 1.2 s each. Nothing was rebuilt. The image does not contain the model, it resolves `models:/taxi-demand@production` at start-up and on `/v1/admin/reload`.

## What did not work, and what we would change

* **The PSI stays high after a promotion.** Each new model keeps the reference distribution of its own training window, and that window still contains the transition. So the drift alarm keeps ringing in April while the model is fine (WAPE 29%, close to its deploy-time 33%). The "does it hurt?" check kept this from causing retraining, but an alarm that is always on gets ignored. **Fix:** rebuild the reference from the most recent stable week after each promotion (a rolling reference).
* **The hurt threshold is relative to deploy-time WAPE.** That is right for a stable regime. At 7 pickups per zone-hour, though, a WAPE of 30% is about 2 pickups, and the business may not care. A floor in absolute MAE would cut useless retraining further.
* **Promotion is a human step, run by `--approve` in the replay.** In real life the 19 March candidate would have waited for a reviewer, perhaps a day. We think that is the right trade. A model that moved every lag by a factor of 5 deserves a human look.

## What to automate next month

1. A nightly job (Airflow or a cron GitHub Action) that runs the drift check, the hurt check and the gated retraining, then opens a ticket with the gate numbers.
2. The `retrain` workflow (`.github/workflows/retrain.yml`, `workflow_dispatch`, environment approval) wired to a shared registry and a DVC remote, so CI can see the same data and models as the laptop.
3. A rolling reference, rebuilt after every promotion.
4. Canary release: send 10% of traffic to `@staging` for a day before the promotion, and compare live WAPE.
