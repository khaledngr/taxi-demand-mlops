"""Integration fixtures: a throwaway MLflow registry with a real model at @production.

No TLC download needed, so CI runs these on every push.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import taxi.model as model_mod
from taxi import common
from taxi.features import build


def synthetic_counts(hours: int = 24 * 21) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-01-01", periods=hours, freq="h")
    rows = []
    for zone, base in [(132, 300), (161, 200), (237, 150)]:
        y = base + 0.5 * base * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 10, hours)
        rows.append(pd.DataFrame({"zone": zone, "ts": ts, "y": np.clip(y, 0, None).round()}))
    return pd.concat(rows)


@pytest.fixture(scope="session")
def registry(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlflow")
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
    feats = tmp / "features.parquet"
    df = build(synthetic_counts(), top_zones=3, ref_end="2020-01-14")
    df.to_parquet(feats)
    mp.setattr(common, "FEATURES_PATH", feats)
    mp.setattr(model_mod, "FEATURES_PATH", feats)
    mp.setattr(common, "STATE_DIR", tmp / "state")

    train = df[df.ts < "2020-01-15"]
    valid = df[df.ts >= "2020-01-15"]
    _, _, uri, run_id = model_mod.train_and_log(train, valid, run_name="test")
    v1 = model_mod.register(uri, run_id, "production")
    yield {"df": df, "version": v1, "tmp": tmp}
    mp.undo()
