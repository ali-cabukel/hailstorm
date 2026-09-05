"""
Cleaning + feature engineering for the NYC TLC yellow-taxi tip-percentage model.

Deliberately free of Ray imports so it can be unit-tested on a plain DataFrame
and reused identically at training and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

# Only these are read off disk. Column pushdown at the Parquet level is the
# single biggest speedup in this pipeline -- the raw files carry ~19 columns.
RAW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "tip_amount",
]

TARGET = "tip_pct"

# NOTE ON LEAKAGE: `total_amount` is excluded on purpose. It is defined as
# fare + extra + mta_tax + tip + tolls + surcharge, so it contains the target.
# Including it gives a near-perfect R^2 and a completely worthless model. This
# is the most common way this dataset gets fumbled.
FORBIDDEN = {"total_amount", "tip_amount", "payment_type", TARGET}

NUMERIC_FEATURES = [
    "trip_distance",
    "duration_min",
    "speed_mph",
    "fare_amount",
    "extra",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "passenger_count",
    "hour_sin",
    "hour_cos",
    "dow",
    "is_weekend",
    "is_night",
    "store_and_fwd",
    "pu_airport",
    "do_airport",
]

# Handed to XGBoost as native categoricals (enable_categorical=True).
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID", "VendorID"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed vocabularies. These MUST be pinned rather than inferred per-batch:
# Ray Data hands each worker a different slice, and if categories are inferred
# locally the integer codes stop meaning the same thing across shards, which
# silently corrupts the model. TLC zones run 1-265.
ZONE_IDS = list(range(1, 266))
RATECODE_IDS = [1, 2, 3, 4, 5, 6, 99]
VENDOR_IDS = [1, 2, 6, 7]

CATEGORY_VOCAB = {
    "PULocationID": ZONE_IDS,
    "DOLocationID": ZONE_IDS,
    "RatecodeID": RATECODE_IDS,
    "VendorID": VENDOR_IDS,
}

# Airport zones: JFK, LaGuardia, Newark.
AIRPORT_ZONES = {132, 138, 1}

# --------------------------------------------------------------------------
# Cleaning thresholds
# --------------------------------------------------------------------------

MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
MIN_DISTANCE_MI = 0.1
MAX_DISTANCE_MI = 100.0
MIN_FARE = 2.5
MAX_FARE = 500.0
MAX_SPEED_MPH = 80.0
MAX_TIP_PCT = 1.0  # tips above 100% of fare are real but rare; treated as noise


def clean_and_featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Filter junk rows and build the model matrix.

    Applied per-batch by Ray Data, so it must be order-independent and must not
    depend on any statistic computed over the full dataset.
    """
    df = df.copy()

    # --- credit card only -------------------------------------------------
    # Cash tips are handed over in the car and never recorded, so cash rows
    # have tip_amount == 0 by construction. Training on them teaches the model
    # to predict "how was this paid", not "how much was tipped".
    df = df[df["payment_type"] == 1]

    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60.0

    # --- physical plausibility -------------------------------------------
    df = df[
        df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
        & df["trip_distance"].between(MIN_DISTANCE_MI, MAX_DISTANCE_MI)
        & df["fare_amount"].between(MIN_FARE, MAX_FARE)
        & (df["tip_amount"] >= 0)
    ]

    df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60.0)
    df = df[df["speed_mph"] <= MAX_SPEED_MPH]

    # --- target -----------------------------------------------------------
    df[TARGET] = df["tip_amount"] / df["fare_amount"]
    df = df[df[TARGET].between(0.0, MAX_TIP_PCT)]

    # --- temporal features ------------------------------------------------
    # Hour is cyclic: 23:00 and 00:00 are adjacent, and a raw integer hides
    # that from a tree that can only split on thresholds.
    pickup = pickup.loc[df.index]
    hour = pickup.dt.hour + pickup.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow"] = pickup.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_night"] = ((pickup.dt.hour >= 22) | (pickup.dt.hour < 6)).astype("int8")

    # Deliberately NOT adding month/year. Because the split is temporal, those
    # take values at test time that were never seen in training, and a tree
    # cannot extrapolate past a split threshold.

    # --- misc -------------------------------------------------------------
    df["store_and_fwd"] = (df["store_and_fwd_flag"] == "Y").astype("int8")
    df["pu_airport"] = df["PULocationID"].isin(AIRPORT_ZONES).astype("int8")
    df["do_airport"] = df["DOLocationID"].isin(AIRPORT_ZONES).astype("int8")

    for col in ["passenger_count", "congestion_surcharge", "airport_fee", "extra", "tolls_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

    # --- categoricals with pinned vocabulary ------------------------------
    for col, vocab in CATEGORY_VOCAB.items():
        codes = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.Categorical(codes, categories=vocab)

    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype("float32")

    out = df[FEATURES + [TARGET]]
    assert not (FORBIDDEN & set(out.columns) - {TARGET}), "leaked column in feature matrix"
    return out.reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].to_numpy(dtype="float32")
