"""Shared helpers: config, paths, git and data fingerprints, MLflow setup."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FEATURES_PATH = ROOT / "data" / "features.parquet"
REFERENCE_PATH = ROOT / "artifacts" / "reference.json"
REPORTS_DIR = ROOT / "reports"
STATE_DIR = ROOT / "state"

FEATURES = ["zone", "hour", "dow", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET = "y"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text())


def params() -> dict:
    return load_yaml("params.yaml")


def policy() -> dict:
    return load_yaml("policy.yaml")


def git_info() -> tuple[str, bool]:
    """Commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", True
    return commit, dirty


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_mlflow() -> None:
    """Local SQLite store by default, so a fresh clone needs no server. Override with MLFLOW_TRACKING_URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
