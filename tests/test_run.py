from pathlib import Path
from types import SimpleNamespace

from hailstorm.nyc_taxi_tips.run import (
    _dashboard_url,
    _existing_cluster,
    _optuna_pid_path,
    _port_open,
    sqlite_url,
)


def test_dashboard_url_from_attr() -> None:
    assert _dashboard_url(SimpleNamespace(dashboard_url="127.0.0.1:8265")) == "http://127.0.0.1:8265"
    assert _dashboard_url(SimpleNamespace(dashboard_url="http://127.0.0.1:8265")) == "http://127.0.0.1:8265"


def test_dashboard_url_from_address_info() -> None:
    info = SimpleNamespace(address_info={"webui_url": "127.0.0.1:8265"})
    assert _dashboard_url(info) == "http://127.0.0.1:8265"


def test_dashboard_url_missing() -> None:
    assert _dashboard_url(SimpleNamespace()) is None


def test_sqlite_url_is_absolute(tmp_path: Path) -> None:
    url = sqlite_url(tmp_path / "optuna.db")
    assert url.startswith("sqlite:///")
    assert url.endswith("/optuna.db")
    assert "///" in url


def test_port_open_closed_port() -> None:
    assert _port_open("127.0.0.1", 1) is False


def test_optuna_pid_path(tmp_path: Path) -> None:
    assert _optuna_pid_path(tmp_path / "optuna.db").name == "optuna.db.dashboard.pid"


def test_existing_cluster_from_job_env(monkeypatch) -> None:
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_JOB_CONFIG_JSON", raising=False)
    monkeypatch.delenv("RAY_JOB_ID", raising=False)
    assert _existing_cluster() is False
    monkeypatch.setenv("RAY_JOB_ID", "raysubmit_test")
    assert _existing_cluster() is True
