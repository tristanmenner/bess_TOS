"""Performance analysis and summary reporting for a TOS run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from helper import DataReport
from validation import ValidationReport


def _state_runs(states: pd.Series, target: str) -> list[int]:
    """Return the lengths (in intervals) of contiguous runs of ``target``."""
    is_target = (states == target).to_numpy()
    runs: list[int] = []
    count = 0
    for flag in is_target:
        if flag:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def _fmt(value, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.4f}{unit}"
    return f"{value}{unit}"


def compute_performance(
    config: Config,
    frame: pd.DataFrame,
    diagnostics: dict,
) -> dict:
    """Compute every headline metric requested for the summary."""
    dt = config.interval_hours
    interval_minutes = config.INTERVAL_MINUTES

    charge_runs = _state_runs(frame["operating_state"], "charge")
    discharge_runs = _state_runs(frame["operating_state"], "discharge")

    def _avg(runs: list[int]) -> float:
        return float(np.mean(runs) * interval_minutes) if runs else 0.0

    def _max(runs: list[int]) -> float:
        return float(max(runs) * interval_minutes) if runs else 0.0

    total_revenue = float(frame["interval_revenue"].sum())

    # ---- daily (calendar-day) performance ---------------------------------
    daily = frame.groupby("day")["interval_revenue"].sum()
    daily = daily.sort_index()
    best_day = daily.idxmax()
    worst_day = daily.idxmin()

    daily_import = frame.groupby("day")["charge_energy_kwh"].sum()

    # ---- rolling 24-hour performance --------------------------------------
    window = int(round(24 * 60 / interval_minutes))
    rolling = frame["interval_revenue"].rolling(window=window).sum()

    return {
        "total_discharge_revenue": float(frame["discharge_revenue"].sum()),
        "total_charge_cost": float(frame["charge_cost"].sum()),
        "total_net_revenue": total_revenue,
        "total_energy_charged_kwh": float(frame["charge_energy_kwh"].sum()),
        "total_energy_discharged_kwh": float(frame["discharge_energy_kwh"].sum()),
        "total_battery_throughput_kwh": float(frame["battery_discharge_kwh"].sum()),
        "equivalent_full_cycles": float(frame["battery_discharge_kwh"].sum() / config.usable_energy_kwh),
        "average_soc_percent": float(frame["soc_percent"].mean()),
        "min_soc_percent": float(frame["soc_percent"].min()),
        "max_soc_percent": float(frame["soc_percent"].max()),
        "final_soc_percent": float(frame["soc_percent"].iloc[-1]),
        "n_charge_events": len(charge_runs),
        "n_discharge_events": len(discharge_runs),
        "avg_charge_duration_min": _avg(charge_runs),
        "avg_discharge_duration_min": _avg(discharge_runs),
        "longest_charge_duration_min": _max(charge_runs),
        "longest_discharge_duration_min": _max(discharge_runs),
        "max_interval_revenue": float(frame["interval_revenue"].max()),
        "min_interval_revenue": float(frame["interval_revenue"].min()),
        "n_days": int(daily.size),
        "avg_daily_revenue": float(daily.mean()),
        "median_daily_revenue": float(daily.median()),
        "best_day_date": str(pd.Timestamp(best_day).date()),
        "best_day_revenue": float(daily.max()),
        "worst_day_date": str(pd.Timestamp(worst_day).date()),
        "worst_day_revenue": float(daily.min()),
        "max_rolling_24h_revenue": float(rolling.max()),
        "min_rolling_24h_revenue": float(rolling.min()),
        "avg_daily_import_kwh": float(daily_import.mean()),
        "max_daily_import_kwh": float(daily_import.max()),
        "diagnostics": diagnostics,
    }


def format_summary(
    config: Config,
    perf: dict,
    data_report: DataReport,
    validation: ValidationReport,
    mode_label: str,
) -> str:
    """Render the full human-readable summary."""
    diag = perf["diagnostics"]
    lines: list[str] = []
    sep = "=" * 78

    lines.append(sep)
    lines.append(f"BESS THEORETICALLY OPTIMAL STRATEGY (TOS) - {mode_label}")
    lines.append("Perfect-hindsight benchmark: NOT deployable trading logic.")
    lines.append(sep)

    lines.append("CONFIGURATION")
    lines.append(f"  region                     : {config.REGION}")
    lines.append(f"  start date                 : {config.START_DATE} (inclusive)")
    lines.append(f"  end date                   : {config.END_DATE} (inclusive)")
    lines.append(f"  optimisation mode          : {config.MODE}  ({mode_label})")
    lines.append(f"  battery capacity           : {config.BATTERY_CAPACITY_KWH:,.2f} kWh")
    lines.append(f"  usable capacity            : {config.usable_energy_kwh:,.2f} kWh")
    lines.append(f"  min / max SOC              : {config.MIN_SOC_PERCENT:.2f}% / {config.MAX_SOC_PERCENT:.2f}%")
    lines.append(f"  initial SOC                : {config.INITIAL_SOC_PERCENT:.2f}%")
    lines.append(f"  final SOC                  : {perf['final_soc_percent']:.2f}% (ends), range {perf['min_soc_percent']:.2f}%-{perf['max_soc_percent']:.2f}%")
    lines.append(f"  nominal voltage            : {config.NOMINAL_VOLTAGE_V:,.2f} V")
    lines.append(f"  nominal capacity           : {config.NOMINAL_CAPACITY_AH:,.2f} Ah (= {config.nominal_energy_from_ah_kwh:,.3f} kWh)")
    lines.append(f"  charge C-rate              : {config.MAX_CHARGE_C_RATE:.3f}C  ({config.c_rate_charge_current_a:,.1f} A)")
    lines.append(f"  discharge C-rate           : {config.MAX_DISCHARGE_C_RATE:.3f}C  ({config.c_rate_discharge_current_a:,.1f} A)")
    lines.append(f"  cable current limit        : {config.MAX_CURRENT_A:,.1f} A")
    lines.append(f"  effective charge current   : {config.max_charge_current_a:,.1f} A"
                 f"  ({'cable-limited' if config.charge_current_limited_by_cable else 'C-rate-limited'})")
    lines.append(f"  effective discharge current: {config.max_discharge_current_a:,.1f} A"
                 f"  ({'cable-limited' if config.discharge_current_limited_by_cable else 'C-rate-limited'})")
    lines.append(f"  max charge power           : {config.max_charge_power_kw:,.2f} kW")
    lines.append(f"  max discharge power        : {config.max_discharge_power_kw:,.2f} kW")
    lines.append(f"  round-trip efficiency      : {config.ROUND_TRIP_EFFICIENCY:.4f}")
    lines.append(f"  charge / discharge eff.    : {config.charge_efficiency:.6f} / {config.discharge_efficiency:.6f} (symmetric sqrt split)")
    lines.append(f"  terminal SOC mode          : {config.TERMINAL_SOC_MODE}")
    if config.MODE == 1:
        lines.append(f"  cycle constraint           : {config.CYCLES_PER_DAY:.3f} EFC per calendar day (discharge side)")
    else:
        lines.append("  cycle constraint           : none (free timing)")
    lines.append(f"  interval                   : {config.INTERVAL_MINUTES:g} min = {config.interval_hours:.6f} h")

    lines.append("")
    lines.append("DATA")
    for line in data_report.summary_lines():
        lines.append("  " + line)

    lines.append("")
    lines.append("PERFORMANCE")
    lines.append(f"  total gross discharge revenue : ${perf['total_discharge_revenue']:,.4f}")
    lines.append(f"  total charging cost           : ${perf['total_charge_cost']:,.4f}")
    lines.append(f"  total net revenue             : ${perf['total_net_revenue']:,.4f}")
    lines.append(f"  total energy charged (grid)   : {perf['total_energy_charged_kwh']:,.4f} kWh")
    lines.append(f"  total energy discharged (grid): {perf['total_energy_discharged_kwh']:,.4f} kWh")
    lines.append(f"  total battery throughput      : {perf['total_battery_throughput_kwh']:,.4f} kWh")
    lines.append(f"  equivalent full cycles        : {perf['equivalent_full_cycles']:,.4f} EFC")
    lines.append(f"  average SOC                   : {perf['average_soc_percent']:.3f}%")
    lines.append(f"  minimum SOC reached           : {perf['min_soc_percent']:.3f}%")
    lines.append(f"  maximum SOC reached           : {perf['max_soc_percent']:.3f}%")
    lines.append(f"  number of charge events       : {perf['n_charge_events']}")
    lines.append(f"  number of discharge events    : {perf['n_discharge_events']}")
    lines.append(f"  average charge duration       : {perf['avg_charge_duration_min']:.1f} min")
    lines.append(f"  average discharge duration    : {perf['avg_discharge_duration_min']:.1f} min")
    lines.append(f"  longest charge duration       : {perf['longest_charge_duration_min']:.1f} min")
    lines.append(f"  longest discharge duration    : {perf['longest_discharge_duration_min']:.1f} min")
    lines.append(f"  max single-interval revenue   : ${perf['max_interval_revenue']:,.4f}")
    lines.append(f"  min single-interval revenue   : ${perf['min_interval_revenue']:,.4f}")

    lines.append("")
    lines.append("MONTHLY GRID ENERGY USAGE (for retailer access-tier selection)")
    lines.append(f"  total energy imported (grid)  : {perf['total_energy_charged_kwh']:,.2f} kWh")
    lines.append(f"  total energy exported (grid)  : {perf['total_energy_discharged_kwh']:,.2f} kWh")
    lines.append(f"  net grid energy balance       : {perf['total_energy_charged_kwh'] - perf['total_energy_discharged_kwh']:,.2f} kWh")
    lines.append(f"  average daily import          : {perf['avg_daily_import_kwh']:,.2f} kWh")
    lines.append(f"  highest daily import          : {perf['max_daily_import_kwh']:,.2f} kWh")

    lines.append("")
    lines.append("DAILY PERFORMANCE")
    lines.append(f"  calendar days in period       : {perf['n_days']}")
    lines.append(f"  average daily revenue         : ${perf['avg_daily_revenue']:,.4f}")
    lines.append(f"  median daily revenue          : ${perf['median_daily_revenue']:,.4f}")
    lines.append(f"  highest calendar day          : {perf['best_day_date']}  ${perf['best_day_revenue']:,.4f}")
    lines.append(f"  lowest calendar day           : {perf['worst_day_date']}  ${perf['worst_day_revenue']:,.4f}")
    lines.append(f"  max rolling 24-hour revenue   : ${perf['max_rolling_24h_revenue']:,.4f}")
    lines.append(f"  min rolling 24-hour revenue   : ${perf['min_rolling_24h_revenue']:,.4f}")
    lines.append("  (calendar day = midnight-to-midnight NEM time; rolling 24h = any 288 consecutive intervals)")

    lines.append("")
    lines.append("OPTIMISATION DIAGNOSTICS")
    lines.append(f"  solver                        : {diag['solver']}")
    lines.append(f"  solver status                 : {diag['solver_status']} ({diag['solver_message']})")
    lines.append(f"  objective value               : ${diag['objective_value']:,.6f}")
    lines.append(f"  optimiser runtime             : {diag['runtime_seconds']:.3f} s")
    lines.append(f"  number of intervals           : {diag['n_intervals']}")
    lines.append(f"  number of variables           : {diag['n_variables']}")
    lines.append(f"  equality constraints          : {diag['n_equality_constraints']}")
    lines.append(f"  inequality constraints        : {diag['n_inequality_constraints']}")
    lines.append(f"  binary variables              : {diag['n_binary_variables']}")
    lines.append(f"  no-simultaneity enforced      : {diag['no_simultaneous_enforced']}")
    lines.append(f"  simultaneous fix-ups          : {diag['simultaneous_intervals_canonicalised']}")
    lines.append(f"  missing intervals             : {data_report.n_missing_intervals}")
    lines.append(f"  globally optimal              : {diag['globally_optimal']}")
    lines.append(f"  constraint violations         : {len(validation.violations)}")
    lines.append(f"  independent validation        : {'PASSED' if validation.passed else 'FAILED'}")
    lines.append(sep)
    return "\n".join(lines)


def save_results(frame: pd.DataFrame, path: Path) -> Path:
    """Write the interval-level schedule to CSV for independent inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


__all__ = ["compute_performance", "format_summary", "save_results"]
