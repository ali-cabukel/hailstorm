"""
Distributed final fit with Ray Train's XGBoostTrainer.

When to reach for this instead of `tune.fit_final`:

  tune.fit_final          data fits in one machine's RAM (roughly <50M rows).
                          Simpler, faster, no inter-worker communication.

  this module             full TLC history, tens of GB of features. Data is
                          sharded across the cluster and never materialized on
                          the driver; workers exchange histograms via Rabit.

Uses the post-2.43 Ray Train API, where you supply `train_loop_per_worker`
rather than the old `label_column` + `params` constructor arguments.
"""

from __future__ import annotations

import ray
import xgboost as xgb
from ray.train import Result, RunConfig, ScalingConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from .features import FEATURES, TARGET
from .tune import EARLY_STOP, MAX_ROUNDS


def _train_loop(config: dict) -> None:
    """Runs once per worker. Each sees only its own shard."""
    train_shard = ray.train.get_dataset_shard("train")
    valid_shard = ray.train.get_dataset_shard("valid")

    # Materialize this worker's shard only. With N workers each holds ~1/N of
    # the data, which is the entire reason this scales past one machine.
    train_df = train_shard.materialize().to_pandas()
    valid_df = valid_shard.materialize().to_pandas()

    dtrain = xgb.QuantileDMatrix(
        train_df[FEATURES], label=train_df[TARGET], enable_categorical=True
    )
    dvalid = xgb.QuantileDMatrix(
        valid_df[FEATURES], label=valid_df[TARGET], enable_categorical=True, ref=dtrain
    )

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        **config["params"],
    }

    xgb.train(
        params,
        dtrain,
        num_boost_round=config.get("num_boost_round", MAX_ROUNDS),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
        # Checkpoints the booster and reports metrics back to the driver.
        # Without it the run produces no retrievable model.
        callbacks=[RayTrainReportCallback()],
    )


def _preflight(num_workers: int, cpus_per_worker: int) -> None:
    """Guard against the Ray Train / Ray Data CPU deadlock.

    Training actors hold their CPUs for the whole run. Ray Data needs CPUs of
    its OWN, concurrently, to read and transform the shards those actors are
    waiting on. If the workers reserve every CPU in the cluster, ingestion can
    never be scheduled and the job hangs forever with no error -- it just sits
    at 0%%. Leave headroom.
    """
    total = ray.cluster_resources().get("CPU", 0)
    reserved = num_workers * cpus_per_worker
    if reserved >= total:
        raise RuntimeError(
            f"{num_workers} workers x {cpus_per_worker} CPUs = {reserved} reserved, "
            f"but the cluster only has {total:.0f} CPUs. Ray Data would be starved "
            f"and the run would hang silently. Leave at least ~20% free."
        )


def train_distributed(
    train_ds: ray.data.Dataset,
    valid_ds: ray.data.Dataset,
    best_config: dict,
    num_workers: int = 4,
    cpus_per_worker: int = 4,
    use_gpu: bool = False,
    storage_path: str | None = None,
) -> Result:
    _preflight(num_workers, cpus_per_worker)
    trainer = XGBoostTrainer(
        _train_loop,
        train_loop_config={"params": best_config, "num_boost_round": MAX_ROUNDS},
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": cpus_per_worker},
        ),
        datasets={"train": train_ds, "valid": valid_ds},
        run_config=RunConfig(name="taxi_tip_final", storage_path=storage_path),
    )
    return trainer.fit()


def load_booster(result: Result) -> xgb.Booster:
    return RayTrainReportCallback.get_model(result.checkpoint)
