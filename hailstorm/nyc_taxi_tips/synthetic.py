"""
Generate fake Parquet files that match the TLC yellow-taxi schema.

Purpose: let you run the whole pipeline end to end -- including the cleaning
filters and the temporal split -- on a laptop or in CI without pulling 60GB
from the public bucket. Replace with `real_month_urls()` from data.py when you
point it at the actual data.

The generator injects the same junk the real files contain (negative fares,
zero-distance trips, reversed timestamps, absurd durations) so the cleaning
code is genuinely exercised rather than run against an unrealistically tidy
fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import RATECODE_IDS, VENDOR_IDS, ZONE_IDS


def _month_frame(year: int, month: int, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    span_s = int((end - start).total_seconds())

    pickup = start + pd.to_timedelta(rng.integers(0, span_s, n_rows), unit="s")
    distance = np.abs(rng.lognormal(0.6, 0.9, n_rows)).round(2)

    hour = pickup.hour.to_numpy()
    # Slower in the middle of the day, faster overnight.
    base_speed = 11.0 + 6.0 * np.cos(2 * np.pi * (hour - 3) / 24.0)
    speed = np.clip(base_speed + rng.normal(0, 2.5, n_rows), 3.0, 45.0)
    duration_min = (distance / speed) * 60.0 + rng.exponential(2.0, n_rows)

    dropoff = pickup + pd.to_timedelta(duration_min * 60.0, unit="s")
    fare = (3.0 + 2.9 * distance + 0.45 * duration_min + rng.normal(0, 1.5, n_rows)).round(2)

    pu = rng.choice(ZONE_IDS, n_rows)
    do = rng.choice(ZONE_IDS, n_rows)
    payment = rng.choice([1, 2, 3, 4], n_rows, p=[0.72, 0.24, 0.02, 0.02])

    # Signal the model should be able to recover: a base rate with real
    # dependence on time of day, weekend, airport trips and trip length, plus
    # a lump of riders who take the default screen preset and a lump who
    # tip nothing at all.
    tip_pct = (
        0.19
        + 0.03 * (pickup.dayofweek.to_numpy() >= 5)
        + 0.04 * ((hour >= 22) | (hour < 6))
        - 0.015 * np.log1p(distance)
        + 0.02 * np.isin(pu, [132, 138, 1])
        + rng.normal(0, 0.05, n_rows)
    )
    preset = rng.random(n_rows) < 0.35
    tip_pct = np.where(preset, rng.choice([0.20, 0.25, 0.30], n_rows), tip_pct)
    tip_pct = np.where(rng.random(n_rows) < 0.10, 0.0, tip_pct)
    tip_pct = np.clip(tip_pct, 0.0, 1.2)

    tip = np.where(payment == 1, (tip_pct * fare).round(2), 0.0)
    tolls = np.where(rng.random(n_rows) < 0.06, rng.uniform(3, 12, n_rows).round(2), 0.0)

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "VendorID": rng.choice(VENDOR_IDS, n_rows).astype("int32"),
            "passenger_count": rng.choice([1, 1, 1, 2, 3, 4, 5], n_rows).astype("float64"),
            "trip_distance": distance,
            "RatecodeID": rng.choice(RATECODE_IDS, n_rows, p=[0.9, 0.04, 0.02, 0.01, 0.02, 0.005, 0.005]).astype("float64"),
            "store_and_fwd_flag": rng.choice(["N", "Y"], n_rows, p=[0.98, 0.02]),
            "PULocationID": pu.astype("int32"),
            "DOLocationID": do.astype("int32"),
            "payment_type": payment.astype("int64"),
            "fare_amount": fare,
            "extra": rng.choice([0.0, 0.5, 1.0, 2.5], n_rows),
            "mta_tax": np.full(n_rows, 0.5),
            "tip_amount": tip,
            "tolls_amount": tolls,
            "improvement_surcharge": np.full(n_rows, 0.3),
            "congestion_surcharge": np.where(rng.random(n_rows) < 0.8, 2.5, 0.0),
            "airport_fee": np.where(np.isin(pu, [132, 138]), 1.75, 0.0),
        }
    )
    df["total_amount"] = (
        df.fare_amount + df.extra + df.mta_tax + df.tip_amount
        + df.tolls_amount + df.improvement_surcharge + df.congestion_surcharge
    ).round(2)

    # --- inject the junk that the real files actually contain -------------
    n_bad = max(1, n_rows // 50)
    bad = rng.choice(n_rows, n_bad, replace=False)
    q = np.array_split(bad, 4)
    df.loc[q[0], "fare_amount"] *= -1                       # negative fares
    df.loc[q[1], "trip_distance"] = 0.0                     # zero-distance
    df.loc[q[2], "tpep_dropoff_datetime"] = df.loc[q[2], "tpep_pickup_datetime"]  # zero duration
    df.loc[q[3], "tpep_dropoff_datetime"] = df.loc[q[3], "tpep_pickup_datetime"] + pd.Timedelta(hours=30)
    df.loc[rng.choice(n_rows, n_rows // 100, replace=False), "passenger_count"] = np.nan

    return df


def write_fixture(
    out_dir: str | Path,
    months: list[tuple[int, int]],
    rows_per_month: int = 40_000,
    seed: int = 0,
) -> list[str]:
    """Write one Parquet file per month, named like the real TLC files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for year, month in months:
        df = _month_frame(year, month, rows_per_month, rng)
        p = out / f"yellow_tripdata_{year}-{month:02d}.parquet"
        df.to_parquet(p, index=False)
        paths.append(str(p))
    return paths
