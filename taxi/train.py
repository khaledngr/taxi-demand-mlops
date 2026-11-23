"""Train the first model on the reference period and register it as @staging.

    uv run python -m taxi.train
Promotion to @production is a separate, human step: `uv run python -m taxi.registry promote`.
"""

from __future__ import annotations

from taxi.common import params
from taxi.model import load_features, register, train_and_log, window


def main() -> None:
    p = params()["periods"]
    df = load_features()
    train = window(df, *p["train"])
    valid = window(df, *p["valid"])
    _, metrics, uri, run_id = train_and_log(train, valid, run_name="reference-train", tags={"reason": "initial"})
    version = register(uri, run_id, "staging")
    print(f"run {run_id}: valid MAE {metrics['mae']:.2f}, RMSE {metrics['rmse']:.2f} -> version {version} @staging")


if __name__ == "__main__":
    main()
