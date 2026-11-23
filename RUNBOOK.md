# Runbook: taxi demand API

**Owner:** the on-call ML engineer (Group 3: Khaled Elnagar).
**Who may roll back:** the on-call engineer, without asking anyone. A rollback is cheap: it moves one alias.
**Who may promote:** a reviewer who has read the gate result. The pipeline never moves `@production` by itself.

Every alert in `monitoring/prometheus/alerts.yml` links to a section of this file.

| Alert | Layer | Severity | First action |
|---|---|---|---|
| ApiDown | 1 infrastructure | page | restart, then roll back if the new model broke start-up |
| SlowRequests | 1 infrastructure | page | check load, roll back if the latency came with a new version |
| InputDrift | 2 input | page | find out whether it hurts before you retrain |
| PredictionDrift | 3 prediction | ticket | same as InputDrift, the scores moved too |

## The two commands to know

```bash
make status          # who holds @production and @staging, their metrics and git commit
make rollback        # @production -> the version it replaced; the API reloads at once (about 1 s)
```

## ApiDown

The `/metrics` endpoint has not answered for 30 s.

1. `curl localhost:8000/health`. A 503 means the process is alive but has no model: the registry is unreachable or the alias points to a broken version.
2. Read the last lines of the API log. Look for a load error.
3. If the failure started right after a promotion, run `make rollback` and restart the API (`make serve`).
4. If no version can be loaded, check `MLFLOW_TRACKING_URI` and that `mlflow.db` and `mlruns/` exist.

## SlowRequests

P95 latency is above the 200 ms budget for 1 minute.

1. Compare with the request rate (Grafana panel "Layer 1: requests per second by status"). A spike in traffic is not a model problem.
2. If it began with a new model version (panel "Model version served"), run `make rollback`.
3. Otherwise scale out: the image is stateless, so start a second container behind the load balancer.

## InputDrift

The PSI of a lag feature over the last 300 requests is above 0.25 (`policy.yaml: psi_act`) for 30 s.

Drift is not a reason to retrain. Only harm is.

1. **Is it real?** Look at the Grafana panel "Layer 2: input PSI per feature". One feature for one minute can be a single noisy client. Every lag at once means demand itself changed.
2. **Does it hurt?** The true counts arrive one hour later. Compare today's WAPE with the WAPE the model had when it was trained (`make status`, or `reports/incident_timeline.csv`). It hurts when the ratio reaches `hurt_ratio` (1.2).
3. **Drift without harm:** write it in the ticket and keep watching. Do nothing else.
4. **Drift with harm:** start a gated retraining, either the nightly job or manually:
   ```bash
   uv run python -m taxi.retrain --now "<now>" --reason "InputDrift <ticket>"
   ```
   The candidate is trained on the last 14 labelled days and must beat production by 5% MAE on the last 3 held-out days. If it fails, nothing is registered and production keeps serving. A refusal is a normal outcome: the gate refused 7 candidates in March 2020 while the regime was still moving.
5. **Candidate in `@staging`:** read the gate numbers in MLflow (`make mlflow-ui`, run tagged `reason=...`), then `make promote`.
6. **After the promotion:** watch WAPE for one day. If it gets worse, `make rollback`.

## Rollback

```bash
make rollback                    # or: uv run python -m taxi.registry rollback --api http://localhost:8000
```

* It moves `@production` back to the previous version, from the `production_history` tag on the registered model.
* It tags the version you left as `rolled_back=true`, so it is not promoted again by mistake.
* It asks the running API to reload. Measured time from command to new version serving: about 1.1 s.
* Nothing is rebuilt. The image does not contain the model.

## Changing a threshold

Thresholds live in `policy.yaml` and `monitoring/prometheus/alerts.yml`. A change is a commit with the reason in the message, reviewed like code.
