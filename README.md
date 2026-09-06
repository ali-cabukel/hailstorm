# Hailstorm

Distributed hyperparameter search with **Ray**, **Optuna**, and **XGBoost**.

The first (and currently only) demonstration is NYC yellow-taxi tip-percentage
prediction on the public TLC dataset: predict `tip_amount / fare_amount` for
credit-card trips.

Verified against Ray 2.58, XGBoost 3.4, Optuna 4.9, optuna-dashboard 0.20,
pandas 3.0.

## Install

Python 3.10–3.13. Packaging is **conda + pip + setuptools**, same idea as
`ml-at-scale/ray-intro`. Do **not** use Poetry — `poetry run` has hung
`ray.init()` on this stack (workers never register).

```bash
conda create -n hailstorm python=3.13 -y
conda activate hailstorm
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e ".[dev]"
```

`pip install -e .` puts the `hailstorm` CLI on `PATH`. `.[dev]` adds pytest.

Use Python 3.12 or 3.13. Do **not** install the full `anaconda` metapackage
into this env — it conflicts with the scientific stack.

`aiohttp` is not optional: Ray's path resolver imports fsspec's `HTTPFileSystem`
unconditionally, and fsspec cannot provide it without aiohttp. Without it you
get `ImportError: cannot import name 'HTTPFileSystem'` even on local files.

The `ray[default]` extra is required for the local dashboard at
`http://127.0.0.1:8265`. `ray[tune]` alone is not enough.

## Run the NYC taxi demo

`conda activate hailstorm`, then from the repo root. Every mode also accepts
`--mode real --trials 40 --cpus-per-trial 4` against the public TLC bucket.

Useful flags: `--hpo-rows`, `--fixture-rows`, `--out`, `--optuna-db`,
`--optuna-port`, `--metrics-export-port`, `--start-head`, `--stop-head`,
`--start-metrics`, `--stop-metrics`.

Equivalent module forms: `python -m hailstorm` or
`python -m hailstorm.nyc_taxi_tips`.

If a previous run was interrupted or `ray.init()` hangs after
`Started a local Ray instance` / `Connected to Ray cluster` (workers never
register):

```bash
unset RAY_ADDRESS
ray stop --force
```

### Headless

No standing cluster. Hailstorm starts a throwaway local Ray instance
(`num_cpus=4`) and Optuna dashboard, then tears them down on exit.
This is the path that already finished HPO on a laptop.

```bash
unset RAY_ADDRESS
hailstorm --mode fixture --trials 12
```

Ray: http://127.0.0.1:8265 · Optuna: http://127.0.0.1:8080 (study
`taxi_tip_hpo`). Do not `ray start` first, and do not set `RAY_ADDRESS`.
Overview → Recent Jobs stays empty — open **Jobs** and **Cluster** instead.

### Headed

Standing Ray head plus Optuna dashboard. Cap CPUs at 4 so this laptop
does not prestart 14 workers that never register.

Terminal 1 — starts both dashboards, then exits. Leave the cluster up.

```bash
unset RAY_ADDRESS
hailstorm --start-head
```

Ray: http://127.0.0.1:8265 · Optuna: http://127.0.0.1:8080

Terminal 2 — attach (reuses those dashboards):

```bash
RAY_ADDRESS=auto hailstorm --mode fixture --trials 12
```

You should see `[ray] attaching` then `[ray] connected`. If it sits on
`Connected to Ray cluster` with no `[ray] connected`, Ctrl-C, run
`hailstorm --stop-head`, and use **Headless**. Overview → Recent Jobs
still stays empty — this is an interactive driver, not a submitted job.

Stop the head and Optuna when finished:

```bash
hailstorm --stop-head
unset RAY_ADDRESS
```

### Submit as a job

Same standing head as **Headed** (`hailstorm --start-head` in terminal 1,
Optuna already up), but the run appears under Overview → Recent Jobs.

```bash
RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265 ray job submit \
  -- python -m hailstorm --mode fixture --trials 12
```

**Do not pass `--working-dir .`.** That flag builds a second runtime env;
workers hang on startup (`Job supervisor actor could not be scheduled`).
`pip install -e .` already put `hailstorm` on the head's Python.

Follow a job:

```bash
ray job list
ray job logs <job_id> --follow
ray job status <job_id>
ray job stop <job_id>
```

Tear down with `hailstorm --stop-head`.

## Monitor

Hailstorm starts Optuna automatically (headless: with the driver; headed /
job-submit: with `--start-head`). URLs are printed as each service starts.
Headless dashboards shut down when the process exits; headed ones stay up
until `hailstorm --stop-head`.

| | URL | What to watch |
|---|---|---|
| **Ray Cluster** | `http://127.0.0.1:8265/#/cluster` | Node CPUs, object store, workers |
| **Ray Jobs** | `http://127.0.0.1:8265/#/jobs` | This driver (job id is printed) and Tune trial tasks |
| **Ray Metrics** | `http://127.0.0.1:8265/#/metrics` | Embedded Grafana (after `--start-metrics`) |
| **Optuna** | `http://127.0.0.1:8080` | Study `taxi_tip_hpo`: TPE proposals, history, importances |
| **Prometheus** | `http://127.0.0.1:9090` | Raw scrapes (`ray_dashboard_api_requests_count_requests_total`) |
| **Grafana** | `http://127.0.0.1:3000` | Ray dashboards (admin / admin) |

**Overview → Recent Jobs** only lists **Submit as a job**. Headless and
headed attach are interactive drivers — use **Jobs** and **Cluster**.

**Optuna stays empty until HPO.** Fixture generation and ingest do not create
Optuna trials. Wait for `[hpo] Optuna + ASHA` in the terminal, then open
study `taxi_tip_hpo` and refresh.

The Optuna UI needs a persisted study. Hailstorm writes `optuna.db` (override
with `--optuna-db` / `--optuna-port`) and starts `optuna-dashboard` against it
as soon as the process begins. After a run you can reopen the same file:

```bash
optuna-dashboard sqlite:///optuna.db
```

## Prometheus and Grafana (Docker)

Do **not** install the unsigned Prometheus / Grafana Mac binaries
(`ray metrics launch-prometheus` hits Gatekeeper). Run them as Docker
images instead. Ray exports metrics; Prometheus scrapes Ray; Grafana
queries Prometheus; Ray Dashboard embeds Grafana when the env vars below
are set (hailstorm sets them by default).

Start the containers **before** or right after the Ray cluster:

```bash
hailstorm --start-metrics
# equivalent: docker compose -f docker-compose.metrics.yml up -d
```

Then headless or headed as usual. Hailstorm pins Ray's scrape port at
**44217** so it does not collide with Optuna on 8080, and points the
dashboard at:

```text
RAY_PROMETHEUS_HOST=http://127.0.0.1:9090
RAY_GRAFANA_HOST=http://127.0.0.1:3000
RAY_GRAFANA_IFRAME_HOST=http://127.0.0.1:3000
```

Prometheus reaches Ray on the Mac via `host.docker.internal`. We do not
mount `/tmp/ray/session_latest` — Docker will not follow that symlink.

After the head is up, hailstorm copies Ray's Grafana JSON into
`metrics/grafana/dashboards/` so Grafana's file provisioner picks them
up (folder **Ray**). Health checks:

```text
http://127.0.0.1:8265/api/prometheus_health
http://127.0.0.1:8265/api/grafana_health
```

Stop the stack (does not stop Ray):

```bash
hailstorm --stop-metrics
```

## What each library is actually doing

| | Role | Why it's needed |
|---|---|---|
| **Ray Data** | Parquet ingest, cleaning, feature engineering | The full history is ~60GB; nothing needs to land on the driver |
| **Optuna** | Proposes configs via TPE | The space is correlated (depth × LR × regularisation), which is where TPE beats random search |
| **ASHA** | Kills bad trials at ~50 rounds instead of 2000 | This is where most of the wall-clock saving comes from |
| **Ray Tune** | Runs trials in parallel, holds data in shared memory | `ray.put` once, every trial reads zero-copy |
| **Ray Train** | Distributed final fit | Only for full-history runs; see below |
| **XGBoost** | The model | Native categoricals, `QuantileDMatrix`, per-iteration eval for ASHA |

## Design decisions worth defending in a write-up

**Target is tip percentage, not tip amount.** Tip amount is largely a
restatement of fare. Percentage is the behavioural question.

**Credit-card trips only.** Cash tips are handed over in the car and never
recorded, so cash rows have `tip_amount == 0` by construction. Including them
teaches the model to predict payment method, not tipping.

**`total_amount` is excluded.** It is defined as fare + extra + mta_tax +
**tip** + tolls + surcharge — it contains the target. Including it gives a
near-perfect R² and a worthless model. This is the most common way this dataset
gets fumbled, and `features.FORBIDDEN` asserts against it.

**The split is temporal, not random.** Train 2023 → validate 2024 Q1 → test
2024 Q2. A random split puts trips from the same driver, hour and block on both
sides. Validation sits strictly between train and test in time, so model
selection never sees the test period even indirectly through early stopping.
Reporting both split styles side by side is a good result in its own right.

**Categorical vocabularies are pinned, not inferred.** Ray hands each worker a
different shard. If categories are inferred per-batch, integer codes stop
meaning the same thing across shards and the model is silently corrupted. Zone
IDs are hard-coded as 1–265.

**No month/year features.** Because the split is temporal, those take values at
test time never seen in training, and a tree cannot extrapolate past a split
threshold.

**Hour is encoded cyclically.** 23:00 and 00:00 are adjacent; a raw integer
hides that from a threshold-splitting tree.

## Two gotchas found while testing this

**1. Ray Train + Ray Data will deadlock silently.** Training actors hold their
CPUs for the entire run, while Ray Data needs CPUs of its *own*, concurrently,
to feed those actors. If workers reserve every CPU, ingestion can never be
scheduled and the job hangs at 0% with no error message. `_preflight()` in
`train_distributed.py` raises instead. Leave ~20% of cluster CPUs free.

**2. ASHA silently no-ops without per-iteration reporting.** If a trial only
reports once at the end, the scheduler has nothing to prune on. The
`_TuneReportCallback` in `tune.py` bridges XGBoost's eval history into Tune
every 25 boosting rounds. You can confirm it's working by checking that trials
terminate at *different* iteration counts in the results table.

## Scaling up

`XGBoostTrainer` is the newer post-2.43 API (`train_loop_per_worker`, not the
old `label_column` + `params` constructor). Most tutorials online still show the
deprecated form.

Rough guide:

- **< 50M rows** — skip `train_distributed` entirely, use `tune.fit_final`. No
  inter-worker communication, simpler, usually faster.
- **Full history** — `train_distributed`, `num_workers` ≈ nodes, GPUs via
  `use_gpu=True` plus `device="cuda"` in params.
- **HPO stage** — keep `--hpo-rows` subsampled. A few million rows gives the
  same *ranking* of configs at a fraction of the cost; the full data only comes
  back for the final fit.

## Layout

```
pyproject.toml                setuptools project + hailstorm CLI entry
setup.py                      setuptools shim
requirements.txt              runtime deps (also declared in pyproject.toml)
docker-compose.metrics.yml    Prometheus + Grafana (host.docker.internal)
metrics/                      scrape config + Grafana provisioning
hailstorm/
  cli.py                      hailstorm
  nyc_taxi_tips/
    features.py               cleaning + FE (no Ray import — unit-testable)
    data.py                   Ray Data ingest, temporal splits
    tune.py                   Optuna TPE + ASHA search, final fit, importances
    train_distributed.py      Ray Train XGBoostTrainer for full-scale runs
    synthetic.py              TLC-schema fixture generator
    run.py                    CLI driver; starts Ray + Optuna dashboards
tests/
  test_features.py            leakage / cleaning
  test_run.py                 dashboard URL + SQLite helpers
```

## Tests

```bash
pytest
```

## Next steps that would strengthen the result

- Report random-split vs temporal-split scores side by side to quantify leakage.
- Compare against the training-mean baseline (already in `metrics`) and a
  ridge regression — GBDT should beat both clearly, and if it doesn't, the
  features are the problem.
- Model the zero-tip mass separately: the target is a spike at 0 plus a
  continuous distribution, so a two-stage classifier + regressor often beats a
  single squared-error fit.
- Add the taxi-zone lookup join for borough-level features.
