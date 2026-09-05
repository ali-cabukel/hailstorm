"""Unit tests for the leakage-sensitive cleaning path. No Ray required."""

from __future__ import annotations

import pandas as pd

from hailstorm.nyc_taxi_tips.features import (
    FEATURES,
    FORBIDDEN,
    TARGET,
    clean_and_featurize,
)


def _row(**overrides) -> dict:
    base = {
        "tpep_pickup_datetime": "2023-01-01 10:00:00",
        "tpep_dropoff_datetime": "2023-01-01 10:20:00",
        "VendorID": 1,
        "passenger_count": 1,
        "trip_distance": 3.0,
        "RatecodeID": 1,
        "store_and_fwd_flag": "N",
        "PULocationID": 132,
        "DOLocationID": 50,
        "payment_type": 1,
        "fare_amount": 12.0,
        "extra": 0.5,
        "tolls_amount": 0.0,
        "congestion_surcharge": 2.5,
        "airport_fee": 1.75,
        "tip_amount": 2.4,
        "total_amount": 19.15,
    }
    base.update(overrides)
    return base


def test_drops_cash_trips() -> None:
    out = clean_and_featurize(pd.DataFrame([_row(), _row(payment_type=2, tip_amount=0)]))
    assert len(out) == 1


def test_excludes_leaky_columns() -> None:
    out = clean_and_featurize(pd.DataFrame([_row()]))
    leaked = FORBIDDEN & (set(out.columns) - {TARGET})
    assert not leaked
    assert "total_amount" not in out.columns
    assert "tip_amount" not in out.columns
    assert "payment_type" not in out.columns


def test_tip_pct_and_airport_flag() -> None:
    out = clean_and_featurize(pd.DataFrame([_row()]))
    assert abs(out[TARGET].iloc[0] - 0.2) < 1e-5
    assert out["pu_airport"].iloc[0] == 1
    assert set(FEATURES).issubset(out.columns)


def test_filters_zero_distance() -> None:
    out = clean_and_featurize(pd.DataFrame([_row(trip_distance=0.0)]))
    assert len(out) == 0
