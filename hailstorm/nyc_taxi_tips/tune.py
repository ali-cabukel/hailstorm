"""
Hyperparameter search: Optuna proposes, ASHA prunes, Ray runs trials in parallel.

Division of labour, since the three libraries overlap and it is easy to end up
using one of them for nothing:

  Optuna  -- decides WHICH configs to try (TPE over a correlated space)
  ASHA    -- decides WHICH trials to kill early, at boosting-round granularity
  Ray     -- decides WHERE trials run, and holds the data in shared memory

The pruning is the part that pays. Each trial reports validation RMSE every
`REPORT_EVERY` boosting rounds, so a bad config dies after ~50 rounds instead
of running all 2000.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import ray
import xgboost as xgb
from optuna.storages import BaseStorage
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from .features import FEATURES, split_xy

METRIC = "valid_rmse"
MODE = "min"
STUDY_NAME = "taxi_tip_hpo"
MAX_ROUNDS = 2000
REPORT_EVERY = 25
EARLY_STOP = 100


class _TuneReportCallback(xgb.callback.TrainingCallback):
    """Bridge XGBoost's per-iteration eval history into Tune.

    Without this, Tune only sees a result when the trial finishes, and ASHA has
    nothing to prune on -- which quietly reduces the scheduler to a no-op.
    """

    def __init__(self, eval_name: str = "valid", metric: str = "rmse"):
        self.eval_name = eval_name
        self.metric = metric

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if (epoch + 1) % REPORT_EVERY and epoch != 0:
            return False
        history = evals_log.get(self.eval_name, {}).get(self.metric)
        if history:
            tune.report({METRIC: float(history[-1]), "boost_round": epoch + 1})
        return False  # never request a stop; ASHA owns that decision


def search_space() -> dict:
    """Ranges chosen for a dataset of this shape: millions of rows, ~20 features,
    a noisy target with a hard floor at zero. Depth and learning rate interact,
    which is exactly the case where TPE beats random search."""
    return {
        "max_depth": tune.randint(4, 13),
        "learning_rate": tune.loguniform(1e-2, 3e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.4, 1.0),
        "min_child_weight": tune.loguniform(1.0, 3e2),
        "reg_lambda": tune.loguniform(1e-2, 1e2),
        "reg_alpha": tune.loguniform(1e-3, 1e1),
        "max_cat_to_onehot": tune.randint(1, 16),
    }


def _trainable(config: dict, data_refs: dict):
    """One trial. Runs in its own Ray worker process."""
    train_df = ray.get(data_refs["train"])
    valid_df = ray.get(data_refs["valid"])

    X_tr, y_tr = split_xy(train_df)
    X_va, y_va = split_xy(valid_df)

    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.QuantileDMatrix(X_va, label=y_va, enable_categorical=True, ref=dtrain)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "nthread": int(config.get("nthread", 2)),
        **{k: v for k, v in config.items() if k != "nthread"},
    }

    xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
        callbacks=[_TuneReportCallback()],
    )


def as_optuna_storage(storage: str | BaseStorage | None) -> BaseStorage | None:
    """Ray's OptunaSearch rejects a URL string — it wants a BaseStorage."""
    if storage is None or isinstance(storage, BaseStorage):
        return storage
    return optuna.storages.RDBStorage(
        url=storage,
        heartbeat_interval=1,
        grace_period=10,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


@dataclass
class TuneResult:
    best_config: dict
    best_rmse: float
    n_trials: int


def run_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    num_samples: int = 30,
    cpus_per_trial: int = 2,
    storage_path: str | None = None,
    optuna_storage: str | None = None,
) -> TuneResult:
    # Put the frames in the object store ONCE. Every trial on the same node
    # then reads them from shared memory with no copy and no re-serialization.
    # Passing DataFrames through the config instead would pickle them per trial.
    data_refs = {"train": ray.put(train_df), "valid": ray.put(valid_df)}

    trainable = tune.with_resources(
        tune.with_parameters(_trainable, data_refs=data_refs),
        {"CPU": cpus_per_trial},
    )

    scheduler = ASHAScheduler(
        metric=METRIC,
        mode=MODE,
        max_t=MAX_ROUNDS // REPORT_EVERY,
        grace_period=2,
        reduction_factor=3,
    )

    tuner = tune.Tuner(
        trainable,
        param_space={**search_space(), "nthread": cpus_per_trial},
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(
                metric=METRIC,
                mode=MODE,
                study_name=STUDY_NAME,
                storage=as_optuna_storage(optuna_storage),
            ),
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=tune.RunConfig(
            name="taxi_tip_hpo",
            storage_path=storage_path or os.path.expanduser("~/ray_results"),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric=METRIC, mode=MODE)
    cfg = {k: v for k, v in best.config.items() if k != "nthread"}
    return TuneResult(
        best_config=cfg,
        best_rmse=float(best.metrics[METRIC]),
        n_trials=len(results),
    )


def fit_final(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[xgb.Booster, dict]:
    """Refit on train with the winning config, then score the held-out test months."""
    X_tr, y_tr = split_xy(train_df)
    X_va, y_va = split_xy(valid_df)
    X_te, y_te = split_xy(test_df)

    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.QuantileDMatrix(X_va, label=y_va, enable_categorical=True, ref=dtrain)
    dtest = xgb.DMatrix(X_te, label=y_te, enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        **config,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )

    pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    resid = pred - y_te

    # Baseline: predict the training mean. If the model cannot beat this by a
    # clear margin, the features are not doing any work.
    baseline = float(np.sqrt(np.mean((y_tr.mean() - y_te) ** 2)))

    metrics = {
        "test_rmse": float(np.sqrt(np.mean(resid**2))),
        "test_mae": float(np.mean(np.abs(resid))),
        "baseline_rmse": baseline,
        "best_iteration": int(booster.best_iteration),
        "n_test": int(len(y_te)),
    }
    metrics["improvement_pct"] = 100 * (1 - metrics["test_rmse"] / baseline)
    return booster, metrics


def top_gains(booster: xgb.Booster, k: int = 12) -> pd.DataFrame:
    score = booster.get_score(importance_type="gain")
    rows = [{"feature": f, "gain": score.get(f, 0.0)} for f in FEATURES]
    return (
        pd.DataFrame(rows)
        .sort_values("gain", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
