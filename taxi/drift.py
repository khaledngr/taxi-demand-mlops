"""Drift check: compare the last window of live data with the reference the production model trained on.

    uv run python -m taxi.drift --end 2020-03-20
PSI per feature (0.10 watch, 0.25 act, from policy.yaml), plus KS and Wasserstein to read the size of
the gap. Works without labels: the inputs and the model's own predictions are enough.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from taxi.common import REPORTS_DIR, policy
from taxi.model import load_alias, load_features, predict

EPS = 1e-4


def psi(ref: dict, current: np.ndarray) -> float:
    """Population Stability Index: sum of (cur - ref) * ln(cur / ref) over the reference decile bins."""
    edges = np.asarray(ref["edges"])
    counts, _ = np.histogram(np.clip(current, edges[0], edges[-1]), bins=edges)
    cur = np.clip(counts / max(counts.sum(), 1), EPS, None)
    base = np.clip(np.asarray(ref["fractions"]), EPS, None)
    return float(np.sum((cur - base) * np.log(cur / base)))


def check(frame: pd.DataFrame, reference: dict) -> dict:
    """frame: live rows with the monitored features (including 'prediction'). Returns one verdict per feature."""
    pol = policy()["drift"]
    out = {}
    for feat in pol["features"]:
        cur = frame[feat].to_numpy(dtype=float)
        ref_sample = np.asarray(reference[feat]["sample"])
        value = psi(reference[feat], cur)
        ks = stats.ks_2samp(ref_sample, cur)
        level = "act" if value >= pol["psi_act"] else "watch" if value >= pol["psi_watch"] else "stable"
        out[feat] = {
            "psi": round(value, 4),
            "ks_d": round(float(ks.statistic), 4),
            "ks_p": float(ks.pvalue),
            "wasserstein": round(float(stats.wasserstein_distance(ref_sample, cur)), 3),
            "level": level,
        }
    return out


def run(end: str, model=None, reference=None, df: pd.DataFrame | None = None) -> dict:
    pol = policy()["drift"]
    if model is None:
        model, _, reference = load_alias("production")
    df = load_features() if df is None else df
    end_ts = pd.Timestamp(end)
    live = df[(df.ts > end_ts - pd.Timedelta(hours=pol["window_hours"])) & (df.ts <= end_ts)]
    live = live.assign(prediction=predict(model, live))
    features = check(live, reference)
    worst = max(features.values(), key=lambda f: f["psi"])
    return {
        "window_end": str(end_ts),
        "rows": len(live),
        "features": features,
        "max_psi": worst["psi"],
        "drifted": [f for f, v in features.items() if v["level"] == "act"],
        "decision": "act" if worst["level"] == "act" else worst["level"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help="end of the window, e.g. 2020-03-20 or '2020-03-20 23:00'")
    a = ap.parse_args()
    report = run(a.end)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"drift_{pd.Timestamp(a.end):%Y%m%d_%H}.json"
    path.write_text(json.dumps(report, indent=2))
    for feat, v in report["features"].items():
        print(f"{feat:<11} PSI {v['psi']:6.3f}  KS D {v['ks_d']:.3f}  W {v['wasserstein']:8.2f}  {v['level']}")
    print(f"decision: {report['decision']}  drifted: {report['drifted']}  -> {path.name}")


if __name__ == "__main__":
    main()
