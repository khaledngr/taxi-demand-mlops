"""Reproduction test: refit a registered version from today's code and data, compare with what MLflow logged.

    uv run python -m taxi.repro_check            # version 1, the reference model
    uv run python -m taxi.repro_check --version 3
Passes when the data hash matches the run's data_md5 and every validation metric is equal within 1e-6.
Warns when the run was trained from a dirty tree or another commit: then the code is not provably the same.
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient

from taxi.common import FEATURES_PATH, git_info, md5, params, setup_mlflow
from taxi.model import evaluate, fit, load_features, window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1")
    a = ap.parse_args()
    setup_mlflow()
    client = MlflowClient()
    mv = client.get_model_version(params()["model"]["name"], a.version)
    run = client.get_run(mv.run_id)
    rp, tags, logged = run.data.params, run.data.tags, run.data.metrics

    commit, dirty = git_info()
    if tags.get("git_dirty") == "True" or tags.get("git_commit") != commit:
        print(f"warning: run trained at {tags.get('git_commit', '?')[:8]} (dirty={tags.get('git_dirty')}), "
              f"now at {commit[:8]} (dirty={dirty})")
    data_ok = md5(FEATURES_PATH) == tags.get("data_md5")
    print(f"data md5 {'matches' if data_ok else 'DIFFERS from'} the run")

    df = load_features()
    train = window(df, rp["train_start"], rp["train_end"])
    valid = window(df, rp["valid_start"], rp["valid_end"])
    metrics = evaluate(fit(train, params()["model"]), valid)
    ok = data_ok
    for k, v in metrics.items():
        diff = abs(v - logged[f"valid_{k}"])
        ok &= diff <= 1e-6
        print(f"valid_{k}: logged {logged[f'valid_{k}']:.6f}  refit {v:.6f}  diff {diff:.1e}")
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
