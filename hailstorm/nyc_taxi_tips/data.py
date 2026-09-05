"""
Ray Data ingestion for the NYC TLC yellow-taxi files.

The split is TEMPORAL, by month. A random split on this dataset leaks badly:
trips from the same driver, the same hour and the same block end up on both
sides, and validation RMSE drops by a wide margin for no real reason. Since
the production question is "predict tomorrow's tips from history", the
evaluation has to be arranged the same way.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import ray

from .features import RAW_COLUMNS, clean_and_featurize

TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def real_month_urls(months: Iterable[tuple[int, int]]) -> list[str]:
    """Public TLC URLs. No auth, no login, updated monthly."""
    return [f"{TLC_BASE}/yellow_tripdata_{y}-{m:02d}.parquet" for y, m in months]


def load_months_pandas(paths: list[str]) -> pd.DataFrame:
    """Local fixture path: no Ray workers. Avoids the laptop Data-CPU deadlock."""
    frames = [pd.read_parquet(p, columns=RAW_COLUMNS) for p in paths]
    return clean_and_featurize(pd.concat(frames, ignore_index=True))


def load_months(paths: list[str], override_num_blocks: int | None = None) -> ray.data.Dataset:
    """Read raw Parquet, then clean and featurize batch-by-batch.

    Column pushdown happens at the Parquet reader, so the ~19 unused raw
    columns are never decoded. `clean_and_featurize` is row-independent, which
    is what lets Ray fan it out over arbitrary shards.
    """
    ds = ray.data.read_parquet(
        paths,
        columns=RAW_COLUMNS,
        override_num_blocks=override_num_blocks,
    )
    return ds.map_batches(clean_and_featurize, batch_format="pandas")


def temporal_split(
    train_months: list[tuple[int, int]],
    valid_months: list[tuple[int, int]],
    test_months: list[tuple[int, int]],
    path_fn=real_month_urls,
    override_num_blocks: int | None = None,
) -> dict[str, ray.data.Dataset]:
    return {
        name: load_months(path_fn(months), override_num_blocks)
        for name, months in [
            ("train", train_months),
            ("valid", valid_months),
            ("test", test_months),
        ]
    }


def fixture_path_fn(fixture_dir: str):
    """Swap-in for `real_month_urls` that points at locally generated files."""

    def _fn(months: Iterable[tuple[int, int]]) -> list[str]:
        return [f"{fixture_dir}/yellow_tripdata_{y}-{m:02d}.parquet" for y, m in months]

    return _fn


def to_pandas_capped(ds: ray.data.Dataset, max_rows: int | None) -> pd.DataFrame:
    """Materialize a Ray Dataset to a local DataFrame, optionally subsampled.

    Used for the HPO stage. Hyperparameter search does not need the full
    dataset -- a few million rows gives you the same ranking of configs at a
    fraction of the cost. The full data comes back for the final fit.
    """
    if max_rows is not None:
        total = ds.count()
        if total > max_rows:
            ds = ds.random_sample(max_rows / total, seed=42)
    return ds.to_pandas()
