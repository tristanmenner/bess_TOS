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

Every configuration value can also be overridden from the command line, e.g.:

    python main.py --capacity-kwh 119 --voltage 743.6 --ah 160 --outdir outputs/119kWh
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import BOUND_TOL_USD, CONFIG, Config
from helper import DataReport, DataValidationError, prepare_price_series
from optimizer import build_schedule_frame, solve_lp_bound, solve_schedule
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


def mode_label(config: Config, mode: int) -> str:
    label = MODE_LABELS[mode]
    if not config.ENFORCE_NO_SIMULTANEOUS:
        label += " [LP relaxation - upper bound]"
    return label


def run_mode(config: Config, tidy: pd.DataFrame, data_report: DataReport, mode: int):
    """Solve, validate and report a single optimisation mode."""
    cfg = replace(config, MODE=mode)
    cfg.validate()

    price_kwh = tidy["price_kwh"].to_numpy()
    day_index = pd.factorize(tidy["day"])[0]

    result = solve_schedule(cfg, price_kwh, day_index)

    # Independent optimality certificate: the LP relaxation must not be below
    # the MILP objective.  (In LP mode the objective already IS the bound.)
    if cfg.COMPUTE_LP_BOUND and cfg.ENFORCE_NO_SIMULTANEOUS:
        bound = solve_lp_bound(cfg, price_kwh, day_index)
        result.diagnostics["lp_bound_revenue"] = bound
        result.diagnostics["bound_gap_absolute"] = bound - float(result.diagnostics["total_revenue"])
        bound_tol = max(BOUND_TOL_USD, 1e-9 * abs(bound))
        if bound + bound_tol < float(result.diagnostics["total_revenue"]):
            raise RuntimeError(
                "Optimality certificate failed: MILP revenue "
                f"{result.diagnostics['total_revenue']:.6f} exceeds the LP relaxation "
                f"bound {bound:.6f}."
            )
    elif not cfg.ENFORCE_NO_SIMULTANEOUS:
        # This run IS the LP relaxation, so its own objective is the bound.
        result.diagnostics["lp_bound_revenue"] = float(-result.diagnostics["objective_value"])
        result.diagnostics["bound_gap_absolute"] = 0.0

    frame = build_schedule_frame(cfg, tidy, result)
    validation = validate_schedule(cfg, frame)
    perf = compute_performance(cfg, frame, result.diagnostics)
    summary = format_summary(cfg, perf, data_report, validation, mode_label(cfg, mode))
    return cfg, frame, result, validation, perf, summary


def _output_paths(config: Config, mode: int) -> tuple[Path, Path]:
    stem = f"results_{config.REGION}_{config.START_DATE}_{config.END_DATE}_mode{mode}"
    csv_path = config.OUTPUT_DIR / f"{stem}.csv"
    txt_path = config.OUTPUT_DIR / f"{stem}_summary.txt"
    return csv_path, txt_path


def write_data_manifest(report: DataReport, output_dir: Path) -> Path | None:
    """Persist the SHA-256 of every input file for reproducibility."""
    if not report.file_hashes:
        return None
    path = output_dir / "data_manifest.sha256"
    lines = [f"{digest}  {name}" for name, digest in sorted(report.file_hashes.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perfect-hindsight BESS arbitrage benchmark (TOS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--region", help="NEM region, e.g. VIC1")
    parser.add_argument("--start", help="START_DATE (inclusive), YYYY-MM-DD")
    parser.add_argument("--end", help="END_DATE (inclusive), YYYY-MM-DD")
    parser.add_argument("--mode", type=int, choices=(1, 2), help="run a single mode only")
    parser.add_argument("--capacity-kwh", type=float, dest="capacity_kwh")
    parser.add_argument("--voltage", type=float, dest="voltage_v")
    parser.add_argument("--ah", type=float, dest="capacity_ah")
    parser.add_argument("--min-soc", type=float, dest="min_soc")
    parser.add_argument("--max-soc", type=float, dest="max_soc")
    parser.add_argument("--initial-soc", type=float, dest="initial_soc")
    parser.add_argument("--charge-c-rate", type=float, dest="charge_c_rate")
    parser.add_argument("--discharge-c-rate", type=float, dest="discharge_c_rate")
    parser.add_argument("--max-current", type=float, dest="max_current_a",
                        help="battery/DC cable current limit (A)")
    parser.add_argument("--phases", type=int, choices=(1, 3),
                        help="AC connection phases (1 or 3)")
    parser.add_argument("--current-per-phase", type=float, dest="max_current_a_per_phase",
                        help="AC connection current limit per phase (A)")
    parser.add_argument("--max-export-kw", type=float, dest="max_export_kw",
                        help="static site export cap (kW); default = connection capacity")
    parser.add_argument("--max-import-kw", type=float, dest="max_import_kw",
                        help="static site import cap (kW); default = connection capacity")
    parser.add_argument("--rte", type=float, dest="rte")
    parser.add_argument("--cycles-per-day", type=float, dest="cycles_per_day")
    parser.add_argument("--terminal-soc", choices=("equal_initial", "free"), dest="terminal_soc")
    parser.add_argument("--cycle-basis", choices=("battery_discharge", "battery_charge", "grid_discharge", "grid_charge"),
                        dest="cycle_basis")
    parser.add_argument("--day-basis", choices=("calendar", "nem_trading"), dest="day_basis")
    parser.add_argument("--interval-minutes", type=float, dest="interval_minutes")
    parser.add_argument("--mip-gap", type=float, dest="mip_gap",
                        help="MILP relative gap target (0 = prove optimality)")
    parser.add_argument("--lp-relaxation", action="store_true",
                        help="solve the LP relaxation (optimistic upper bound, not physical)")
    parser.add_argument("--incomplete-data", action="store_true",
                        help="warn instead of failing when the price series does not cover the period")
    parser.add_argument("--no-lp-bound", action="store_true",
                        help="skip the LP upper-bound certificate")
    parser.add_argument("--outdir", help="directory for CSV/summary/plots")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if args.region is not None:
        overrides["REGION"] = args.region
    if args.start is not None:
        overrides["START_DATE"] = args.start
    if args.end is not None:
        overrides["END_DATE"] = args.end
    if args.capacity_kwh is not None:
        overrides["BATTERY_CAPACITY_KWH"] = args.capacity_kwh
    if args.voltage_v is not None:
        overrides["NOMINAL_VOLTAGE_V"] = args.voltage_v
    if args.capacity_ah is not None:
        overrides["NOMINAL_CAPACITY_AH"] = args.capacity_ah
    if args.min_soc is not None:
        overrides["MIN_SOC_PERCENT"] = args.min_soc
    if args.max_soc is not None:
        overrides["MAX_SOC_PERCENT"] = args.max_soc
    if args.initial_soc is not None:
        overrides["INITIAL_SOC_PERCENT"] = args.initial_soc
    if args.charge_c_rate is not None:
        overrides["MAX_CHARGE_C_RATE"] = args.charge_c_rate
    if args.discharge_c_rate is not None:
        overrides["MAX_DISCHARGE_C_RATE"] = args.discharge_c_rate
    if args.max_current_a is not None:
        overrides["MAX_CURRENT_A"] = args.max_current_a
    if args.phases is not None:
        overrides["PHASES"] = args.phases
    if args.max_current_a_per_phase is not None:
        overrides["MAX_CURRENT_A_PER_PHASE"] = args.max_current_a_per_phase
    if args.max_export_kw is not None:
        overrides["MAX_EXPORT_KW"] = args.max_export_kw
    if args.max_import_kw is not None:
        overrides["MAX_IMPORT_KW"] = args.max_import_kw
    if args.rte is not None:
        overrides["ROUND_TRIP_EFFICIENCY"] = args.rte
    if args.cycles_per_day is not None:
        overrides["CYCLES_PER_DAY"] = args.cycles_per_day
    if args.terminal_soc is not None:
        overrides["TERMINAL_SOC_MODE"] = args.terminal_soc
    if args.cycle_basis is not None:
        overrides["MODE1_CYCLE_BASIS"] = args.cycle_basis
    if args.day_basis is not None:
        overrides["DAY_BASIS"] = args.day_basis
    if args.interval_minutes is not None:
        overrides["INTERVAL_MINUTES"] = args.interval_minutes
    if args.mip_gap is not None:
        overrides["MIP_REL_GAP"] = args.mip_gap
    if args.lp_relaxation:
        overrides["ENFORCE_NO_SIMULTANEOUS"] = False
    if args.incomplete_data:
        overrides["REQUIRE_FULL_COVERAGE"] = False
    if args.no_lp_bound:
        overrides["COMPUTE_LP_BOUND"] = False
    if args.outdir is not None:
        outdir = Path(args.outdir).resolve()
        overrides["OUTPUT_DIR"] = outdir
        overrides["PLOT_PATH"] = outdir / "generated_plot.png"

    config = replace(CONFIG, **overrides)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NEM DISPATCHPRICE data from {config.DATA_DIR} ...")
    try:
        tidy, data_report = prepare_price_series(config)
    except DataValidationError as exc:
        print(f"\nDATA ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: fix the input archive, adjust START_DATE/END_DATE, or pass "
            "--incomplete-data to accept (and report) partial coverage.",
            file=sys.stderr,
        )
        return 2
    print(f"  prepared {len(tidy):,} intervals for {config.REGION} "
          f"({tidy['day'].nunique()} {config.DAY_BASIS} days)")
    for warning in data_report.warnings:
        print(f"  WARNING: {warning}")
    manifest = write_data_manifest(data_report, config.OUTPUT_DIR)
    if manifest is not None:
        print(f"  input file hashes written to {manifest}")

    modes = (args.mode,) if args.mode is not None else RUN_MODES
    summaries: list[str] = []
    dashboards: dict[int, pd.DataFrame] = {}
    perfs: dict[int, dict] = {}

    for mode in modes:
        cfg, frame, result, validation, perf, summary = run_mode(
            config, tidy, data_report, mode
        )
        dashboards[mode] = frame
        perfs[mode] = perf

        csv_path, txt_path = _output_paths(config, mode)
        save_results(frame, csv_path)
        txt_path.write_text(summary + "\n", encoding="utf-8")
        summaries.append(summary)

        print(f"\nMode {mode}: solved in {result.diagnostics['runtime_seconds']:.3f}s, "
              f"net revenue ${perf['total_net_revenue']:,.2f}, "
              f"validated={validation.passed}, saved {csv_path.name}")

    # ---- plots -------------------------------------------------------------
    def break_even_args(mode: int) -> dict:
        """Amber subscription line for this mode's own grid import."""
        perf = perfs[mode]
        cost = float(perf["amber_period_cost"])
        months = int(perf["amber_billing_months"])
        month_label = f"{months} month" + ("" if months == 1 else "s")
        return {
            "break_even_cost": cost,
            "break_even_label": (
                f"Amber subscription: ${cost:,.2f} "
                f"({perf['amber_tier_label']}, {month_label})"
            ),
        }

    primary_mode = modes[0]
    primary_frame = dashboards[primary_mode]
    plot_dashboard(
        primary_frame,
        replace(config, MODE=primary_mode),
        config.PLOT_PATH,
        **break_even_args(primary_mode),
    )
    plot_cumulative_energy(
        primary_frame,
        replace(config, MODE=primary_mode),
        config.PLOT_PATH.parent / "generated_energy_plot.png",
    )
    for mode, frame in dashboards.items():
        plot_cumulative_revenue(
            frame,
            replace(config, MODE=mode),
            config.OUTPUT_DIR / f"cumulative_revenue_mode{mode}.png",
            **break_even_args(mode),
        )
        plot_cumulative_energy(
            frame,
            replace(config, MODE=mode),
            config.OUTPUT_DIR / f"cumulative_energy_mode{mode}.png",
        )
        if mode != primary_mode:
            plot_dashboard(
                frame,
                replace(config, MODE=mode),
                config.OUTPUT_DIR / f"dashboard_mode{mode}.png",
                **break_even_args(mode),
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
