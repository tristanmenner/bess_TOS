"""BESS Theoretically Optimal Strategy (TOS) - main entry point.

Pipeline (see README):
    configuration -> data discovery -> loading -> validation -> preprocessing
    -> price series -> optimisation -> constraint validation -> revenue
    -> performance analysis -> plotting -> reporting

Run with:
    pip install -r requirements.txt
    python main.py

The strategy is a perfect-hindsight benchmark. It uses the complete historical
price series to compute the maximum achievable arbitrage revenue. It is NOT a
forecasting or deployable trading strategy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import CONFIG
from helper import DataReport, prepare_price_series
from optimizer import build_schedule_frame, solve_schedule
from reporting import compute_performance, format_summary, save_results
from validation import validate_schedule
from visualization import (
    plot_cumulative_energy,
    plot_cumulative_revenue,
    plot_dashboard,
)

# The approved run executes both optimisation modes for the configured period.
RUN_MODES = (2, 1)

MODE_LABELS = {
    1: "Mode 1 - fixed daily cycle count",
    2: "Mode 2 - unconstrained timing (free arbitrage)",
}


def run_mode(config, tidy: pd.DataFrame, data_report: DataReport, mode: int):
    """Solve, validate and report a single optimisation mode."""
    cfg = replace(config, MODE=mode)
    cfg.validate()

    price_kwh = tidy["price_kwh"].to_numpy()
    day_index = pd.factorize(tidy["day"])[0]

    result = solve_schedule(cfg, price_kwh, day_index)
    frame = build_schedule_frame(cfg, tidy, result)
    validation = validate_schedule(cfg, frame)
    perf = compute_performance(cfg, frame, result.diagnostics)
    summary = format_summary(cfg, perf, data_report, validation, MODE_LABELS[mode])
    return cfg, frame, result, validation, perf, summary


def _output_paths(config, mode: int) -> tuple[Path, Path]:
    stem = f"results_{config.REGION}_{config.START_DATE}_{config.END_DATE}_mode{mode}"
    csv_path = config.OUTPUT_DIR / f"{stem}.csv"
    txt_path = config.OUTPUT_DIR / f"{stem}_summary.txt"
    return csv_path, txt_path


def main() -> int:
    config = CONFIG
    config.validate()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NEM DISPATCHPRICE data from {config.DATA_DIR} ...")
    tidy, data_report = prepare_price_series(config)
    print(f"  prepared {len(tidy):,} intervals for {config.REGION} "
          f"({tidy['day'].nunique()} calendar days)")
    for warning in data_report.warnings:
        print(f"  WARNING: {warning}")

    summaries: list[str] = []
    dashboards: dict[int, pd.DataFrame] = {}

    for mode in RUN_MODES:
        cfg, frame, result, validation, perf, summary = run_mode(
            config, tidy, data_report, mode
        )
        dashboards[mode] = frame

        csv_path, txt_path = _output_paths(config, mode)
        save_results(frame, csv_path)
        txt_path.write_text(summary + "\n", encoding="utf-8")
        summaries.append(summary)

        print(f"\nMode {mode}: solved in {result.diagnostics['runtime_seconds']:.3f}s, "
              f"net revenue ${perf['total_net_revenue']:,.2f}, "
              f"validated={validation.passed}, saved {csv_path.name}")

    # ---- plots -------------------------------------------------------------
    primary_frame = dashboards[RUN_MODES[0]]
    plot_dashboard(primary_frame, replace(config, MODE=RUN_MODES[0]), config.PLOT_PATH)
    plot_cumulative_revenue(
        primary_frame,
        replace(config, MODE=RUN_MODES[0]),
        config.OUTPUT_DIR / f"cumulative_revenue_mode{RUN_MODES[0]}.png",
    )
    plot_cumulative_energy(
        primary_frame,
        replace(config, MODE=RUN_MODES[0]),
        config.PLOT_PATH.parent / "generated_energy_plot.png",
    )
    for mode in RUN_MODES[1:]:
        plot_dashboard(
            dashboards[mode],
            replace(config, MODE=mode),
            config.OUTPUT_DIR / f"dashboard_mode{mode}.png",
        )
    for mode, frame in dashboards.items():
        plot_cumulative_energy(
            frame,
            replace(config, MODE=mode),
            config.OUTPUT_DIR / f"cumulative_energy_mode{mode}.png",
        )

    print()
    for summary in summaries:
        print(summary)
        print()
    print(f"Plot written to: {config.PLOT_PATH}")
    print(f"Outputs written to: {config.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
