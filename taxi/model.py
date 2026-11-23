"""Training, evaluation, drift reference and MLflow logging, shared by train.py and retrain.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from taxi.common import FEATURES, FEATURES_PATH, TARGET, git_info, md5, params, policy, setup_mlflow

REFERENCE_SAMPLE = 2000
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= ts <= end (end inclusive, whole day if only a date is given)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end == end.normalize():
        end = end + pd.Timedelta(hours=23)
    return df[(df.ts >= start) & (df.ts <= end)]


def fit(train: pd.DataFrame, mp: dict) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=mp["n_estimators"],
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        random_state=mp["random_state"],
    )
    model.fit(train[FEATURES], train[TARGET])
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(df[FEATURES]), 0, None)  # a pickup count is never negative


def evaluate(model, df: pd.DataFrame) -> dict:
    pred = predict(model, df)
    return {
        "mae": float(mean_absolute_error(df[TARGET], pred)),
        "rmse": float(root_mean_squared_error(df[TARGET], pred)),
    }


def build_reference(model, train: pd.DataFrame) -> dict:
    """What 'normal' looks like for the drift checks: decile edges plus a sample, per monitored feature."""
    frame = train.assign(prediction=predict(model, train))
    ref = {}
    for feat in policy()["drift"]["features"]:
        values = frame[feat].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        rng = np.random.default_rng(0)
        sample = rng.choice(values, size=min(REFERENCE_SAMPLE, len(values)), replace=False)
        ref[feat] = {"edges": edges.tolist(), "fractions": (counts / counts.sum()).tolist(), "sample": sample.tolist()}
    return ref


def train_and_log(train: pd.DataFrame, valid: pd.DataFrame, run_name: str, tags: dict | None = None):
    """Fit, evaluate on the held-out window, log everything to MLflow. Returns (model, metrics, model_uri, run_id)."""
    setup_mlflow()
    p = params()
    mp = p["model"]
    mlflow.set_experiment("taxi-demand")
    commit, dirty = git_info()
    with mlflow.start_run(run_name=run_name) as run:
        model = fit(train, mp)
        metrics = evaluate(model, valid)
        mlflow.log_params({k: v for k, v in mp.items() if k != "name"})
        mlflow.log_params(
            {
                "train_start": str(train.ts.min()),
                "train_end": str(train.ts.max()),
                "valid_start": str(valid.ts.min()),
                "valid_end": str(valid.ts.max()),
                "n_train": len(train),
                "top_zones": p["data"]["top_zones"],
            }
        )
        mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()})
        mlflow.set_tags(
            {"git_commit": commit, "git_dirty": str(dirty), "data_md5": md5(FEATURES_PATH), **(tags or {})}
        )
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.json"
            ref_path.write_text(json.dumps(build_reference(model, train)))
            mlflow.log_artifact(str(ref_path))
        # skops format (no pickle); the one tree type it flags is ours, trained in this run, so we trust it
        info = mlflow.sklearn.log_model(
            model, name="model", input_example=train[FEATURES].head(3), skops_trusted_types=TRUSTED_TYPES
        )
    return model, metrics, info.model_uri, run.info.run_id


def register(model_uri: str, run_id: str, alias: str) -> int:
    """Register a logged model as a new version and point the alias at it."""
    name = params()["model"]["name"]
    version = mlflow.register_model(model_uri, name, tags={"run_id": run_id})
    MlflowClient().set_registered_model_alias(name, alias, version.version)
    return int(version.version)


def load_alias(alias: str = "production"):
    """Load a model by name and alias (never by path or run id). Returns (model, version, reference)."""
    setup_mlflow()
    name = params()["model"]["name"]
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    run_id = mv.tags.get("run_id") or mv.run_id
    with tempfile.TemporaryDirectory() as tmp:
        ref = json.loads(Path(client.download_artifacts(run_id, "reference.json", tmp)).read_text())
    return model, int(mv.version), ref
