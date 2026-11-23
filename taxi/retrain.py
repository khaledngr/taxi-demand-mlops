"""Retraining with a gate. The alarm only asks; the gate decides.

    uv run python -m taxi.retrain --now 2020-03-25 --reason "psi=0.9 on lag_24"

1. cooldown: at most one run per policy.retrain.cooldown_hours (state/last_retrain.json)
2. candidate trained on the most recent labelled days (train_days), same code as train.py
3. gate: candidate and production both scored on the last gate_days, held out from the candidate
4. pass (MAE at least min_gain lower) -> registered, @staging; a human then runs `registry promote`
   fail -> nothing registered, production stays as it is
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from taxi.common import STATE_DIR, policy
from taxi.model import evaluate, load_alias, load_features, register, train_and_log

STATE = STATE_DIR / "last_retrain.json"


def hurts(recent_mae: float, baseline_mae: float) -> bool:
    """Drift alone is not a reason to retrain: retrain only if the error grew past hurt_ratio x baseline."""
    return recent_mae >= policy()["retrain"]["hurt_ratio"] * baseline_mae


def cooldown_over(now: pd.Timestamp) -> bool:
    if not STATE.exists():
        return True
    last = pd.Timestamp(json.loads(STATE.read_text())["at"])
    return now - last >= pd.Timedelta(hours=policy()["retrain"]["cooldown_hours"])


def run(now: str, reason: str = "manual", force: bool = False, df: pd.DataFrame | None = None) -> dict:
    pol = policy()["retrain"]
    now_ts = pd.Timestamp(now)
    if not force and not cooldown_over(now_ts):
        return {"status": "skipped", "why": "cooldown"}

    df = load_features() if df is None else df
    labelled = df[df.ts < now_ts]  # a pickup count is known one hour later, so everything before now
    gate_start = now_ts - pd.Timedelta(days=pol["gate_days"])
    train_start = gate_start - pd.Timedelta(days=pol["train_days"])
    train = labelled[(labelled.ts >= train_start) & (labelled.ts < gate_start)]
    gate = labelled[labelled.ts >= gate_start]

    prod_model, prod_version, _ = load_alias("production")
    prod = evaluate(prod_model, gate)
    _, cand, uri, run_id = train_and_log(
        train, gate, run_name=f"retrain-{now_ts:%Y%m%d}", tags={"reason": reason, "trigger_time": str(now_ts)}
    )
    gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    passed = gain >= pol["min_gain"]
    result = {
        "status": "promoted_to_staging" if passed else "refused_by_gate",
        "now": str(now_ts),
        "production_version": prod_version,
        "production_mae": round(prod["mae"], 2),
        "candidate_mae": round(cand["mae"], 2),
        "gain": round(gain, 3),
        "min_gain": pol["min_gain"],
        "run_id": run_id,
    }
    if passed:
        result["staging_version"] = register(uri, run_id, "staging")
    STATE_DIR.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"at": str(now_ts), **result}))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", required=True)
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--force", action="store_true", help="ignore the cooldown")
    a = ap.parse_args()
    print(json.dumps(run(a.now, a.reason, a.force), indent=2))


if __name__ == "__main__":
    main()
