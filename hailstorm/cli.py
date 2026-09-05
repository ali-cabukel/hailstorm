"""Hailstorm CLI — currently the NYC taxi tip-percentage demo."""


def main() -> None:
    # Importing Ray / XGBoost takes 10–30s with no other output. Say so first.
    print("hailstorm: loading Ray, Optuna, XGBoost ...", flush=True)
    from hailstorm.nyc_taxi_tips.run import main as run_main

    run_main()
