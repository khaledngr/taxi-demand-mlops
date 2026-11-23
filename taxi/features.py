"""Aggregate trips to one row per (zone, hour) and add lag and calendar features."""

from __future__ import annotations

import pandas as pd

from taxi.common import FEATURES_PATH, RAW_DIR, params

LAGS = {"lag_1": 1, "lag_2": 2, "lag_3": 3, "lag_24": 24, "lag_168": 168}


def hourly_counts(months: list[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        df = pd.read_parquet(
            RAW_DIR / f"yellow_tripdata_{month}.parquet", columns=["tpep_pickup_datetime", "PULocationID"]
        )
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)
        # TLC files contain a few rows with wrong years: keep only this month
        df = df[(df.tpep_pickup_datetime >= start) & (df.tpep_pickup_datetime < end)]
        df["ts"] = df.tpep_pickup_datetime.dt.floor("h")
        frames.append(df.groupby(["PULocationID", "ts"]).size().rename("y").reset_index())
    counts = pd.concat(frames).rename(columns={"PULocationID": "zone"})
    return counts


def build(counts: pd.DataFrame, top_zones: int, ref_end: str) -> pd.DataFrame:
    # busiest zones measured on the reference (training) period only, so no future data leaks in
    ref = counts[counts.ts <= ref_end]
    zones = ref.groupby("zone").y.sum().nlargest(top_zones).index
    counts = counts[counts.zone.isin(zones)]

    # complete grid: an hour with no pickup is a 0, not a missing row
    hours = pd.date_range(counts.ts.min(), counts.ts.max(), freq="h")
    grid = pd.MultiIndex.from_product([zones, hours], names=["zone", "ts"]).to_frame(index=False)
    df = grid.merge(counts, on=["zone", "ts"], how="left").fillna({"y": 0})
    df = df.sort_values(["zone", "ts"]).reset_index(drop=True)

    for name, k in LAGS.items():
        df[name] = df.groupby("zone").y.shift(k)
    df["hour"] = df.ts.dt.hour
    df["dow"] = df.ts.dt.dayofweek
    df = df.dropna().reset_index(drop=True)
    int_cols = ["zone", "hour", "dow", "y", *LAGS]
    df[int_cols] = df[int_cols].astype("int64")
    return df


def main() -> None:
    p = params()
    counts = hourly_counts(p["data"]["months"])
    df = build(counts, p["data"]["top_zones"], p["periods"]["train"][1])
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"{len(df):,} rows, {df.zone.nunique()} zones, {df.ts.min()} to {df.ts.max()} -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
