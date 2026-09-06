"""
Driver: ingest -> temporal split -> Optuna/ASHA search -> final fit -> test score.

    hailstorm --mode fixture --trials 12
    hailstorm --mode real --trials 40 --cpus-per-trial 4

`fixture` mode generates synthetic TLC-shaped Parquet locally and needs no
network. `real` mode pulls the public files from the TLC bucket.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna
import ray

from . import data as data_mod
from .synthetic import write_fixture
from .tune import STUDY_NAME, fit_final, run_search, top_gains

# Train on 2023, validate on early 2024, test on late 2024. Validation sits
# strictly between train and test in time, so model selection never sees data
# from the test period -- not even indirectly through early stopping.
TRAIN_MONTHS = [(2023, m) for m in range(1, 13)]
VALID_MONTHS = [(2024, m) for m in (1, 2, 3)]
TEST_MONTHS = [(2024, m) for m in (4, 5, 6)]

# Cut down so fixture mode finishes in a minute or two.
FIXTURE_TRAIN = [(2023, m) for m in (1, 4, 7, 10)]
FIXTURE_VALID = [(2024, 1)]
FIXTURE_TEST = [(2024, 4)]

# Optuna uses 8080. Ray's metrics demo often uses 8080 too — do not collide.
DEFAULT_METRICS_EXPORT_PORT = 44217
DEFAULT_PROMETHEUS_HOST = "http://127.0.0.1:9090"
DEFAULT_GRAFANA_HOST = "http://127.0.0.1:3000"


def _dashboard_url(info) -> str | None:
    """Ray returns `127.0.0.1:8265` or a full URL depending on version."""
    url = getattr(info, "dashboard_url", None)
    if not url:
        address_info = getattr(info, "address_info", None) or {}
        url = address_info.get("webui_url")
    if not url:
        return None
    return url if str(url).startswith("http") else f"http://{url}"


def sqlite_url(db_path: Path) -> str:
    """Absolute SQLite URL. Three slashes plus an absolute path is four slashes."""
    return f"sqlite:///{db_path.resolve()}"


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _optuna_pid_path(db_path: Path) -> Path:
    return Path(str(db_path) + ".dashboard.pid")


def _ensure_optuna_study(storage: str) -> None:
    rdb = optuna.storages.RDBStorage(
        url=storage,
        heartbeat_interval=1,
        grace_period=10,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    optuna.create_study(
        storage=rdb,
        study_name=STUDY_NAME,
        direction="minimize",
        load_if_exists=True,
    )


def _start_optuna_dashboard(
    storage: str,
    db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    persistent: bool = False,
) -> subprocess.Popen | None:
    """Serve the persisted study so trials appear live during the sweep."""
    _ensure_optuna_study(storage)
    if _port_open(host, port):
        _log(f"[optuna] dashboard already running at http://{host}:{port}  "
             f"(study '{STUDY_NAME}')")
        return None
    exe = shutil.which("optuna-dashboard")
    if not exe:
        _log("[optuna] dashboard not available — install with: pip install -e .")
        return None
    try:
        proc = subprocess.Popen(
            [exe, storage, "--host", host, "--port", str(port), "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        _log(f"[optuna] dashboard failed to start ({exc})")
        return None
    time.sleep(0.4)
    if proc.poll() is not None:
        _log(f"[optuna] dashboard exited immediately (is port {port} free?)")
        return None
    if persistent:
        _optuna_pid_path(db_path).write_text(str(proc.pid))
    else:
        atexit.register(proc.terminate)
    _log(f"[optuna] dashboard http://{host}:{port}  (open study '{STUDY_NAME}')")
    _log("[optuna] trials appear only after HPO starts — the study is empty until then")
    return proc


def _stop_optuna_dashboard(db_path: Path) -> None:
    pid_path = _optuna_pid_path(db_path)
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        _log(f"[optuna] stopped dashboard (pid {pid})")
    except (ValueError, ProcessLookupError, PermissionError) as exc:
        _log(f"[optuna] could not stop dashboard ({exc})")
    pid_path.unlink(missing_ok=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _compose_file() -> Path:
    return _repo_root() / "docker-compose.metrics.yml"


def _configure_metrics_env() -> None:
    """Tell Ray Dashboard where Docker Prometheus / Grafana live (host URLs)."""
    os.environ.setdefault("RAY_PROMETHEUS_HOST", DEFAULT_PROMETHEUS_HOST)
    os.environ.setdefault("RAY_GRAFANA_HOST", DEFAULT_GRAFANA_HOST)
    os.environ.setdefault(
        "RAY_GRAFANA_IFRAME_HOST",
        os.environ.get("RAY_GRAFANA_HOST", DEFAULT_GRAFANA_HOST),
    )
    os.environ.setdefault("RAY_PROMETHEUS_NAME", "Prometheus")
    for key in (
        "RAY_PROMETHEUS_HOST",
        "RAY_GRAFANA_HOST",
        "RAY_GRAFANA_IFRAME_HOST",
    ):
        os.environ[key] = os.environ[key].rstrip("/")


def _log_metrics_urls() -> None:
    prom = os.environ.get("RAY_PROMETHEUS_HOST", DEFAULT_PROMETHEUS_HOST)
    graf = os.environ.get("RAY_GRAFANA_IFRAME_HOST") or os.environ.get(
        "RAY_GRAFANA_HOST", DEFAULT_GRAFANA_HOST,
    )
    _log(f"[metrics] Prometheus {prom}  (query ray_dashboard_api_requests_count_requests_total)")
    _log(f"[metrics] Grafana    {graf}  (admin / admin)")
    if not _port_open("127.0.0.1", 9090):
        _log("[metrics] Prometheus is not up — start with: hailstorm --start-metrics")


def _copy_ray_grafana_dashboards() -> None:
    src = Path("/tmp/ray/session_latest/metrics/grafana/dashboards")
    dst = _repo_root() / "metrics" / "grafana" / "dashboards"
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in src.glob("*.json"):
        shutil.copy2(path, dst / path.name)
        copied += 1
    if copied:
        _log(f"[metrics] copied {copied} Ray Grafana dashboard(s) to {dst}")


def _docker_compose(*args: str) -> int:
    compose = _compose_file()
    if not compose.is_file():
        _log(f"[metrics] missing {compose}")
        return 1
    docker = shutil.which("docker")
    if not docker:
        _log("[metrics] docker not on PATH")
        return 1
    cmd = [docker, "compose", "-f", str(compose), *args]
    return subprocess.call(cmd)


def _start_metrics() -> None:
    _log("[metrics] starting Prometheus + Grafana (Docker)")
    code = _docker_compose("up", "-d")
    if code != 0:
        raise SystemExit(code)
    _log("[metrics] Prometheus http://127.0.0.1:9090")
    _log("[metrics] Grafana    http://127.0.0.1:3000  (admin / admin)")


def _stop_metrics() -> None:
    code = _docker_compose("down")
    if code != 0:
        raise SystemExit(code)
    _log("[metrics] Prometheus + Grafana stopped")


def _start_head(args: argparse.Namespace) -> None:
    """Standing Ray head + Optuna dashboard (headed / job-submit)."""
    _configure_metrics_env()
    db_path = Path(args.optuna_db)
    storage = sqlite_url(db_path)
    _start_optuna_dashboard(
        storage, db_path, port=args.optuna_port, persistent=True,
    )
    ray_exe = shutil.which("ray") or "ray"
    subprocess.check_call([
        ray_exe, "start", "--head",
        "--dashboard-host=127.0.0.1",
        "--num-cpus=4",
        f"--metrics-export-port={args.metrics_export_port}",
        "--disable-usage-stats",
    ])
    _copy_ray_grafana_dashboards()
    _log("[ray] dashboard http://127.0.0.1:8265")
    _log(f"[optuna] dashboard http://127.0.0.1:{args.optuna_port}")
    _log_metrics_urls()
    _log("[ray] headed cluster is up — attach with RAY_ADDRESS=auto hailstorm ...")


def _stop_head(args: argparse.Namespace) -> None:
    ray_exe = shutil.which("ray") or "ray"
    subprocess.call([ray_exe, "stop", "--force"])
    _stop_optuna_dashboard(Path(args.optuna_db))
    _log("[ray] headed cluster stopped")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _existing_cluster() -> bool:
    """True when this process should attach, not spawn a new head."""
    return bool(
        os.environ.get("RAY_ADDRESS")
        or os.environ.get("RAY_JOB_CONFIG_JSON")
        or os.environ.get("RAY_JOB_ID")
    )


def _init_ray(metrics_export_port: int = DEFAULT_METRICS_EXPORT_PORT):
    """Standalone: start a local head + dashboard. Job / RAY_ADDRESS: attach."""
    _configure_metrics_env()
    if _existing_cluster():
        _log("[ray] attaching to the running cluster (no new dashboard)")
        _log("[ray] if this sits on 'Connected' with no [ray] connected line,")
        _log("[ray]   Ctrl-C, then:  unset RAY_ADDRESS && hailstorm --mode fixture --trials 12")
        return ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            namespace="hailstorm",
        )
    _log("[ray] starting local cluster (num_cpus=4, avoids 14 hung prestart workers) ...")
    try:
        info = ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=True,
            dashboard_host="127.0.0.1",
            _metrics_export_port=metrics_export_port,
        )
        _copy_ray_grafana_dashboards()
        return info
    except Exception as exc:
        _log(f"[ray] dashboard failed to start ({exc}); continuing without it")
        if ray.is_initialized():
            ray.shutdown()
        return ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
            _metrics_export_port=metrics_export_port,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="hailstorm",
        description="NYC taxi tip-percentage demo: Ray + Optuna + XGBoost.",
    )
    ap.add_argument("--mode", choices=["fixture", "real"], default="fixture")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--cpus-per-trial", type=int, default=1)
    ap.add_argument("--hpo-rows", type=int, default=150_000,
                    help="Rows subsampled for the search stage; None-ish for all.")
    ap.add_argument("--fixture-rows", type=int, default=40_000)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--optuna-db", type=str, default="optuna.db",
                    help="SQLite file for the Optuna study (required by optuna-dashboard).")
    ap.add_argument("--optuna-port", type=int, default=8080)
    ap.add_argument("--metrics-export-port", type=int,
                    default=DEFAULT_METRICS_EXPORT_PORT,
                    help="Ray Prometheus scrape port (not 8080 — Optuna).")
    ap.add_argument("--start-head", action="store_true",
                    help="Start a standing Ray head and Optuna dashboard, then exit.")
    ap.add_argument("--stop-head", action="store_true",
                    help="Stop the standing Ray head and Optuna dashboard.")
    ap.add_argument("--start-metrics", action="store_true",
                    help="Start Prometheus + Grafana via docker-compose.metrics.yml.")
    ap.add_argument("--stop-metrics", action="store_true",
                    help="Stop the Prometheus + Grafana containers.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    exclusive = [
        name for name, on in (
            ("--start-head", args.start_head),
            ("--stop-head", args.stop_head),
            ("--start-metrics", args.start_metrics),
            ("--stop-metrics", args.stop_metrics),
        ) if on
    ]
    if len(exclusive) > 1:
        _log("hailstorm: use only one of " + ", ".join(exclusive))
        raise SystemExit(2)
    if args.start_metrics:
        _start_metrics()
        return
    if args.stop_metrics:
        _stop_metrics()
        return
    if args.start_head:
        _start_head(args)
        return
    if args.stop_head:
        _stop_head(args)
        return

    try:
        _run(args)
    except KeyboardInterrupt:
        _log("\nhailstorm: interrupted — shutting down Ray")
        if ray.is_initialized():
            ray.shutdown()
        raise SystemExit(130) from None


def _run(args: argparse.Namespace) -> None:
    optuna_storage = sqlite_url(Path(args.optuna_db))
    _start_optuna_dashboard(
        optuna_storage, Path(args.optuna_db), port=args.optuna_port,
    )

    if args.mode == "fixture":
        tmp = tempfile.mkdtemp(prefix="taxi_fixture_")
        months = FIXTURE_TRAIN + FIXTURE_VALID + FIXTURE_TEST
        _log(f"[fixture] writing {len(months)} months x {args.fixture_rows:,} rows ...")
        write_fixture(tmp, months, rows_per_month=args.fixture_rows)
        path_fn = data_mod.fixture_path_fn(tmp)
        train_m, valid_m, test_m = FIXTURE_TRAIN, FIXTURE_VALID, FIXTURE_TEST
        _log(f"[fixture] wrote {len(months)} synthetic months to {tmp}")
    else:
        path_fn = data_mod.real_month_urls
        train_m, valid_m, test_m = TRAIN_MONTHS, VALID_MONTHS, TEST_MONTHS
        _log("[data] using public TLC parquet URLs (this will download)")

    info = _init_ray(args.metrics_export_port)
    _log("[ray] connected")
    _log_metrics_urls()
    dash = _dashboard_url(info)
    if dash:
        base = dash.rstrip("/")
        try:
            job_id = ray.get_runtime_context().get_job_id()
        except Exception:
            job_id = "?"
        _log(f"[ray] dashboard {base}")
        _log(f"[ray]   cluster  {base}/#/cluster")
        _log(f"[ray]   jobs     {base}/#/jobs")
        _log(f"[ray]   job id   {job_id}")
        if not _existing_cluster():
            _log("[ray] Overview > Recent Jobs stays empty for a local script "
                 "(that list is only `ray job submit`). Use Cluster + Jobs.")
    else:
        _log("[ray] dashboard not available — install with: pip install -e .")

    if args.mode == "fixture":
        _log("[data] cleaning fixture in-process (a few seconds) ...")
        _log("[data]   train months ...")
        train_df = data_mod.load_months_pandas(path_fn(train_m))
        _log("[data]   valid months ...")
        valid_df = data_mod.load_months_pandas(path_fn(valid_m))
        _log("[data]   test months ...")
        test_df = data_mod.load_months_pandas(path_fn(test_m))
        if args.hpo_rows and len(train_df) > args.hpo_rows:
            train_df = train_df.sample(args.hpo_rows, random_state=42)
        if args.hpo_rows and len(valid_df) > args.hpo_rows:
            valid_df = valid_df.sample(args.hpo_rows, random_state=42)
    else:
        _log("[data] ingest + clean via Ray Data ...")
        splits = data_mod.temporal_split(train_m, valid_m, test_m, path_fn=path_fn)
        train_df = data_mod.to_pandas_capped(splits["train"], args.hpo_rows)
        valid_df = data_mod.to_pandas_capped(splits["valid"], args.hpo_rows)
        test_df = splits["test"].to_pandas()
    _log(f"[data] train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")
    _log(f"[data] mean tip pct: train={train_df.tip_pct.mean():.4f} test={test_df.tip_pct.mean():.4f}")

    _log(f"[hpo] Optuna + ASHA, {args.trials} trials — dashboards update from here")
    res = run_search(
        train_df, valid_df,
        num_samples=args.trials,
        cpus_per_trial=args.cpus_per_trial,
        optuna_storage=optuna_storage,
    )
    _log(f"\n[hpo] {res.n_trials} trials, best valid RMSE {res.best_rmse:.5f}")
    for k, v in sorted(res.best_config.items()):
        _log(f"       {k:22s} {v}")

    _log("[fit] refitting winner on train, scoring test ...")
    booster, metrics = fit_final(train_df, valid_df, test_df, res.best_config)
    _log("\n[test]")
    for k, v in metrics.items():
        _log(f"       {k:18s} {v:.5f}" if isinstance(v, float) else f"       {k:18s} {v}")

    _log("\n[importance]")
    _log(top_gains(booster).to_string(index=False))

    out = Path(args.out)
    out.write_text(json.dumps({"best_config": res.best_config, "metrics": metrics}, indent=2))
    booster.save_model(str(out.with_suffix(".ubj")))
    _log(f"\nwrote {out} and {out.with_suffix('.ubj')}")


if __name__ == "__main__":
    main()
