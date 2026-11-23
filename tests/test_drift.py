from __future__ import annotations

import numpy as np

from taxi.drift import check, psi

REF = {"edges": [0, 10, 20, 30, 40], "fractions": [0.25, 0.25, 0.25, 0.25]}


def test_psi_is_zero_for_the_same_distribution():
    same = np.array([5, 15, 25, 35] * 50, dtype=float)
    assert psi(REF, same) < 1e-6


def test_psi_matches_the_hand_computation():
    # live shares 0.10, 0.20, 0.30, 0.40 against 0.25 each
    cur = np.array([5] * 10 + [15] * 20 + [25] * 30 + [35] * 40, dtype=float)
    expected = sum((c - 0.25) * np.log(c / 0.25) for c in [0.1, 0.2, 0.3, 0.4])
    assert abs(psi(REF, cur) - expected) < 1e-9


def test_collapse_in_demand_is_flagged_act(registry):
    import pandas as pd

    from taxi.model import load_alias, predict

    model, _, ref = load_alias("production")
    df = registry["df"]
    normal = df[df.ts >= "2020-01-15"].copy()
    collapsed = normal.copy()
    for col in ["lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "y"]:
        collapsed[col] = (collapsed[col] * 0.1).round()  # a COVID-like 90% drop

    def frame(d: pd.DataFrame) -> pd.DataFrame:
        return d.assign(prediction=predict(model, d))

    assert all(v["level"] != "act" for v in check(frame(normal), ref).values())
    flagged = check(frame(collapsed), ref)
    assert flagged["lag_1"]["level"] == "act" and flagged["prediction"]["level"] == "act"


def test_drift_alone_does_not_retrain():
    from taxi.retrain import hurts

    assert not hurts(recent_wape=0.12, baseline_wape=0.11)  # drifted inputs, same error: keep watching
    assert hurts(recent_wape=0.20, baseline_wape=0.11)  # error up by more than hurt_ratio (1.2x): retrain
