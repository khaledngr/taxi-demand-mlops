"""Replay real zone-hours against the running API, for the live demo.

    uv run python -m taxi.traffic normal          # February 2020: what the model was trained on
    uv run python -m taxi.traffic covid           # late March 2020: demand has collapsed
Watch taxi_input_psi in Prometheus / Grafana move from 'stable' to above 0.25, then the alert fire.
"""

from __future__ import annotations

import argparse
import time

import requests

from taxi.model import load_features

PERIODS = {"normal": ("2020-02-17", "2020-02-24"), "covid": ("2020-03-23", "2020-03-30")}
LAGS = ["lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("period", choices=sorted(PERIODS))
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=600, help="number of requests")
    ap.add_argument("--rate", type=float, default=20, help="requests per second")
    a = ap.parse_args()

    df = load_features()
    start, end = PERIODS[a.period]
    rows = df[(df.ts >= start) & (df.ts < end)].sample(a.n, replace=True, random_state=1)
    errors, abs_err = 0, 0.0
    for i, r in enumerate(rows.itertuples(), 1):
        body = {"zone": int(r.zone), "ts": r.ts.isoformat(), **{k: int(getattr(r, k)) for k in LAGS}}
        resp = requests.post(f"{a.api}/v1/predict", json=body, timeout=5)
        if resp.ok:
            abs_err += abs(resp.json()["predicted_pickups"] - r.y)  # the true count, as if it arrived an hour later
        else:
            errors += 1
        if i % 100 == 0:
            print(f"{i} requests, {errors} errors, MAE so far {abs_err / max(i - errors, 1):.1f}")
        time.sleep(1 / a.rate)


if __name__ == "__main__":
    main()
