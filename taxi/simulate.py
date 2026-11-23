"""Incident simulation: replay the live period (March to April 2020) one day at a time, as production would.

    uv run python -m taxi.simulate --approve
Every night: score the day with the @production model (the true counts arrive an hour later, so the MAE
is known), run the drift check on the last 24 hours, and when the PSI says 'act' and the cooldown is over,
start a gated retraining. A candidate that passes the gate lands in @staging; --approve plays the human
who moves it to @production (without it, the run stops at @staging, as in real life).
Writes reports/incident_timeline.csv, reports/incident_timeline.png and reports/incident_summary.json.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from taxi import drift, registry, retrain
from taxi.common import REPORTS_DIR, params
from taxi.model import evaluate, load_alias, load_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="act as the reviewer: promote what passes the gate")
    ap.add_argument("--start", default=params()["periods"]["live"][0])
    ap.add_argument("--end", default=params()["periods"]["live"][1])
    a = ap.parse_args()

    retrain.STATE.unlink(missing_ok=True)  # a fresh replay starts with no cooldown
    df = load_features()
    model, version, reference = load_alias("production")
    frozen, frozen_version, _ = model, version, reference  # what would happen if nobody reacted
    rows, events = [], []

    for day in pd.date_range(a.start, a.end, freq="D"):
        today = df[(df.ts >= day) & (df.ts < day + pd.Timedelta(days=1))]
        if today.empty:
            continue
        mae = evaluate(model, today)["mae"]
        frozen_mae = evaluate(frozen, today)["mae"]
        check = drift.run(str(day + pd.Timedelta(hours=23)), model, reference, df)
        row = {
            "day": day.date(),
            "production_version": version,
            "mae_production": round(mae, 2),
            "mae_if_no_reaction": round(frozen_mae, 2),
            "actual_mean_pickups": round(today.y.mean(), 1),
            "max_psi": check["max_psi"],
            "drift_decision": check["decision"],
            "drifted_features": ",".join(check["drifted"]),
            "action": "",
        }
        if check["decision"] == "act":
            res = retrain.run(str(day + pd.Timedelta(days=1)), reason=f"psi={check['max_psi']:.2f} {check['drifted']}",
                              df=df)
            row["action"] = res["status"]
            if res["status"] != "skipped":
                events.append({"day": str(day.date()), "max_psi": check["max_psi"], **res})
                print(f"{day.date()} PSI {check['max_psi']:.2f} -> retrain: {res['status']} "
                      f"(prod MAE {res['production_mae']}, candidate {res['candidate_mae']}, gain {res['gain']:+.1%})")
            if res["status"] == "promoted_to_staging" and a.approve:
                version = registry.promote()
                model, _, reference = load_alias("production")
                row["action"] += f" -> approved, v{version} @production"
                events[-1]["approved_to_production"] = version
        rows.append(row)

    timeline = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    timeline.to_csv(REPORTS_DIR / "incident_timeline.csv", index=False)
    plot(timeline)
    first_alarm = timeline[timeline.drift_decision == "act"].day.min()
    summary = {
        "period": [a.start, a.end],
        "initial_production_version": frozen_version,
        "final_production_version": version,
        "first_psi_alarm": str(first_alarm),
        "retraining_events": events,
        "mean_mae_with_reaction": round(timeline.mae_production.mean(), 2),
        "mean_mae_if_no_reaction": round(timeline.mae_if_no_reaction.mean(), 2),
    }
    (REPORTS_DIR / "incident_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "retraining_events"}, indent=2, default=str))


def plot(t: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    days = pd.to_datetime(t.day)
    ax1.plot(days, t.actual_mean_pickups, color="black")
    ax1.set_ylabel("mean pickups\nper zone-hour")
    ax1.set_title("NYC yellow taxi, March to April 2020: the drift, the alarm and the reaction")
    ax2.plot(days, t.max_psi, color="tab:red")
    for y, label in [(0.10, "watch"), (0.25, "act")]:
        ax2.axhline(y, ls="--", color="grey", lw=0.8)
        ax2.text(days.iloc[0], y, f" {label}", va="bottom", fontsize=8, color="grey")
    ax2.set_yscale("symlog", linthresh=0.1)
    ax2.set_ylabel("max PSI\n(last 24 h)")
    ax3.plot(days, t.mae_if_no_reaction, label="no reaction (frozen v1)", color="tab:grey", ls="--")
    ax3.plot(days, t.mae_production, label="our system (gated retraining)", color="tab:blue")
    ax3.set_ylabel("MAE\n(pickups)")
    ax3.legend()
    for ax in (ax1, ax2, ax3):
        for d in days[t.action.str.contains("approved", na=False)]:
            ax.axvline(d, color="tab:green", lw=1, alpha=0.7)
        for d in days[t.action.str.contains("refused", na=False)]:
            ax.axvline(d, color="tab:orange", lw=1, alpha=0.5, ls=":")
    fig.text(0.01, 0.005, "green: candidate passed the gate and was approved   orange dotted: refused by the gate",
             fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "incident_timeline.png", dpi=150)


if __name__ == "__main__":
    main()
