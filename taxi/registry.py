"""Promotion and rollback: both only move the @production alias. No file is copied, nothing is rebuilt.

    uv run python -m taxi.registry status
    uv run python -m taxi.registry promote            # @staging -> @production (the human approval step)
    uv run python -m taxi.registry rollback           # @production -> the version it replaced
    add --api http://localhost:8000 to make the running API reload the alias at once
"""

from __future__ import annotations

import argparse
import time

import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from taxi.common import params, setup_mlflow

HISTORY_TAG = "production_history"  # versions that held @production before, oldest first


def _client() -> tuple[MlflowClient, str]:
    setup_mlflow()
    return MlflowClient(), params()["model"]["name"]


def _alias_version(client: MlflowClient, name: str, alias: str) -> int | None:
    try:
        return int(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def _history(client: MlflowClient, name: str) -> list[int]:
    raw = client.get_registered_model(name).tags.get(HISTORY_TAG, "")
    return [int(v) for v in raw.split(",") if v]


def _reload(api: str | None) -> None:
    if api:
        r = requests.post(f"{api}/v1/admin/reload", timeout=30)
        print(f"API reload: {r.status_code} {r.json()}")


def promote(api: str | None = None) -> int:
    client, name = _client()
    new = _alias_version(client, name, "staging")
    if new is None:
        raise SystemExit("nothing in @staging to promote")
    old = _alias_version(client, name, "production")
    if old == new:
        print(f"version {new} is already @production")
        return new
    if old is not None:
        client.set_registered_model_tag(name, HISTORY_TAG, ",".join(map(str, [*_history(client, name), old])))
    client.set_registered_model_alias(name, "production", new)
    client.delete_registered_model_alias(name, "staging")
    print(f"@production: {old} -> {new}")
    _reload(api)
    return new


def rollback(api: str | None = None) -> int:
    t0 = time.time()
    client, name = _client()
    history = _history(client, name)
    if not history:
        raise SystemExit("no previous production version to roll back to")
    current, previous = _alias_version(client, name, "production"), history[-1]
    client.set_registered_model_alias(name, "production", previous)
    client.set_registered_model_tag(name, HISTORY_TAG, ",".join(map(str, history[:-1])))
    if current is not None:
        client.set_model_version_tag(name, str(current), "rolled_back", "true")
    print(f"ROLLBACK @production: {current} -> {previous} in {time.time() - t0:.2f}s")
    _reload(api)
    return previous


def status() -> None:
    client, name = _client()
    holders = {alias: _alias_version(client, name, alias) for alias in ("production", "staging")}
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        run = client.get_run(mv.tags.get("run_id") or mv.run_id)
        aliases = ",".join(f"@{a}" for a, v in holders.items() if v == int(mv.version))
        if mv.tags.get("rolled_back") == "true":
            aliases += " (rolled back)"
        aliases = aliases.strip() or "-"
        print(
            f"v{mv.version:<3} {aliases:<22} valid_mae={run.data.metrics.get('valid_mae', float('nan')):7.2f}"
            f"  reason={run.data.tags.get('reason', '-'):<28} commit={run.data.tags.get('git_commit', '?')[:8]}"
        )
    print(f"production history: {_history(client, name)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "promote", "rollback"])
    ap.add_argument("--api", help="base URL of the running API, to reload the model after the change")
    a = ap.parse_args()
    {"status": status, "promote": lambda: promote(a.api), "rollback": lambda: rollback(a.api)}[a.action]()


if __name__ == "__main__":
    main()
