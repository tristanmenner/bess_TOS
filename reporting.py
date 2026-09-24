"""Performance analysis and summary reporting for a TOS run."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import BOUND_TOL_USD, Config
from helper import DataReport
from validation import ValidationReport

# Amber Electric subscription tiers: (annual usage cap in kWh, monthly fee in
# AUD, label).  The cap is inclusive and the final tier is open-ended.
AMBER_SUBSCRIPTION_TIERS: tuple[tuple[float, float, str], ...] = (
    (10_000.0, 25.0, "up to 10,000 kWh/yr"),
    (20_000.0, 50.0, "up to 20,000 kWh/yr"),
    (50_000.0, 125.0, "up to 50,000 kWh/yr"),
    (100_000.0, 200.0, "up to 100,000 kWh/yr"),
    (math.inf, 350.0, "over 100,000 kWh/yr"),
)


def _calendar_months(start_date: str, end_date: str) -> int:
    """Number of calendar months touched by an inclusive date range.

    A partial month counts as a full month: Amber bills a flat monthly fee, so
    e.g. 2026-08-01..2026-08-31 is 1 month and 2026-08-15..2026-09-14 is 2.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must not precede start_date")
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def amber_subscription_estimate(
    imported_kwh: float, period_days: float, billing_months: int = 1
) -> dict:
    """Estimate the Amber Electric subscription cost for a modelled period.

    The usage bands are *annual*, so the period's grid import is annualised
    (``imported_kwh * 365.25 / period_days``) before the tier lookup.  The cost
    is the flat monthly fee times the number of calendar months billed (a
    partial month counts as a full fee).  With no site load modelled, grid
    import (battery charging) is the usage metric.  This is a break-even
    reference only — it is not deducted from the benchmark revenue.
    """
    period_days = float(period_days)
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    billing_months = int(billing_months)
    if billing_months < 1:
        raise ValueError("billing_months must be at least 1")
    annual_usage_kwh = float(imported_kwh) * 365.25 / period_days
    for cap, fee, label in AMBER_SUBSCRIPTION_TIERS:
        if annual_usage_kwh <= cap:
            return {
                "annual_usage_kwh": annual_usage_kwh,
                "monthly_fee": float(fee),
                "tier_label": label,
                "billing_months": billing_months,
                "period_cost": float(fee) * billing_months,
            }
    raise AssertionError("unreachable: the final tier cap is infinite")


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


def _fmt_optional(value, fmt: str = ",.6f") -> str:
    if value is None:
        return "n/a"
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        pass
    return f"{value:{fmt}}"


def _fmt_cap(value: float | None) -> str:
    """Render an optional static grid import/export cap."""
    return "none (connection capacity)" if value is None else f"{value:,.2f} kW"


def compute_performance(
    config: Config,
    frame: pd.DataFrame,
    diagnostics: dict,
) -> dict:
    """Compute every headline metric requested for the summary."""
    interval_minutes = config.INTERVAL_MINUTES

    charge_runs = _state_runs(frame["operating_state"], "charge")
    discharge_runs = _state_runs(frame["operating_state"], "discharge")

    def _avg(runs: list[int]) -> float:
        return float(np.mean(runs) * interval_minutes) if runs else 0.0

    def _max(runs: list[int]) -> float:
        return float(max(runs) * interval_minutes) if runs else 0.0

    total_revenue = float(frame["interval_revenue"].sum())

    # ---- daily (per Config.DAY_BASIS) performance -------------------------
    intervals_per_day = int(round(24 * 60 / interval_minutes))
    day_counts = frame.groupby("day").size()
    daily = frame.groupby("day")["interval_revenue"].sum()
    daily = daily.sort_index()
    best_day = daily.idxmax()
    worst_day = daily.idxmin()

    daily_import = frame.groupby("day")["charge_energy_kwh"].sum()

    # ---- retail subscription break-even (Amber Electric) -------------------
    period_days = len(frame) * config.interval_hours / 24.0
    billing_months = _calendar_months(config.START_DATE, config.END_DATE)
    amber = amber_subscription_estimate(
        float(frame["charge_energy_kwh"].sum()), period_days, billing_months
    )

    # ---- rolling 24-hour performance --------------------------------------
    window = intervals_per_day
    rolling = frame["interval_revenue"].rolling(window=window).sum()

    n_simultaneous = int((frame["operating_state"] == "simultaneous").sum())

    return {
        "total_discharge_revenue": float(frame["discharge_revenue"].sum()),
        "total_charge_cost": float(frame["charge_cost"].sum()),
        "total_net_revenue": total_revenue,
        "total_energy_charged_kwh": float(frame["charge_energy_kwh"].sum()),
        "total_energy_discharged_kwh": float(frame["discharge_energy_kwh"].sum()),
        "total_battery_charge_kwh": float(frame["battery_charge_kwh"].sum()),
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
        "intervals_per_day": intervals_per_day,
        "n_partial_days": int((day_counts != intervals_per_day).sum()),
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
        "amber_period_days": period_days,
        "amber_billing_months": amber["billing_months"],
        "amber_annual_usage_kwh": amber["annual_usage_kwh"],
        "amber_tier_label": amber["tier_label"],
        "amber_monthly_fee": amber["monthly_fee"],
        "amber_period_cost": amber["period_cost"],
        "n_simultaneous_intervals": n_simultaneous,
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
    if not config.ENFORCE_NO_SIMULTANEOUS:
        lines.append("*** LP RELAXATION - OPTIMISTIC UPPER BOUND, NOT A DISPATCH PLAN ***")
        lines.append("*** Simultaneous charge/discharge is allowed: physically impossible. ***")
    else:
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
    lines.append(f"  DC charge power rating     : {config.dc_charge_power_kw:,.2f} kW (battery side)")
    lines.append(f"  DC discharge power rating  : {config.dc_discharge_power_kw:,.2f} kW (battery side)")
    lines.append(f"  meter-side charge limit    : {config.max_charge_power_kw:,.2f} kW (= DC rating / eta_c)")
    lines.append(f"  meter-side discharge limit : {config.max_discharge_power_kw:,.2f} kW (= DC rating x eta_d)")
    lines.append(f"  grid connection            : {config.PHASES}-phase, "
                 f"{config.MAX_CURRENT_A_PER_PHASE:,.1f} A/phase, "
                 f"{config.NOMINAL_AC_VOLTAGE_PER_PHASE_V:,.1f} V/phase")
    lines.append(f"  connection capacity        : {config.connection_power_kw:,.2f} kW"
                 f" (= phases x A/phase x V/phase)")
    lines.append(f"  static export cap          : {_fmt_cap(config.MAX_EXPORT_KW)}")
    lines.append(f"  static import cap          : {_fmt_cap(config.MAX_IMPORT_KW)}")
    lines.append(f"  effective charge limit     : {config.effective_charge_power_kw:,.2f} kW"
                 f" ({'connection-limited' if config.charge_limited_by_grid else 'battery/converter-limited'})")
    lines.append(f"  effective discharge limit  : {config.effective_discharge_power_kw:,.2f} kW"
                 f" ({'connection-limited' if config.discharge_limited_by_grid else 'battery/converter-limited'})")
    lines.append(f"  round-trip efficiency      : {config.ROUND_TRIP_EFFICIENCY:.4f}")
    lines.append(f"  charge / discharge eff.    : {config.charge_efficiency:.6f} / {config.discharge_efficiency:.6f} (symmetric sqrt split)")
    lines.append(f"  terminal SOC mode          : {config.TERMINAL_SOC_MODE}")
    lines.append(f"  day basis                  : {config.DAY_BASIS}")
    if config.MODE == 1:
        lines.append(f"  cycle constraint           : {config.CYCLES_PER_DAY:.3f} EFC per day (at most)")
        lines.append(f"  cycle basis                : {config.mode1_cycle_basis_label}")
    else:
        lines.append("  cycle constraint           : none (free timing)")
    lines.append(f"  interval                   : {config.INTERVAL_MINUTES:g} min = {config.interval_hours:.6f} h")
    lines.append(f"  no-simultaneity enforced   : {config.ENFORCE_NO_SIMULTANEOUS}")
    lines.append(f"  MILP relative gap target    : {config.MIP_REL_GAP}")
    lines.append(f"  require full data coverage : {config.REQUIRE_FULL_COVERAGE}")

    lines.append("")
    lines.append("DATA (independent of the optimiser)")
    for line in data_report.summary_lines():
        lines.append("  " + line)
    if data_report.n_missing_intervals and not config.REQUIRE_FULL_COVERAGE:
        lines.append("  NOTE: the price series does not cover the whole requested period;")
        lines.append("        every total below is for the covered intervals only.")

    lines.append("")
    lines.append("PERFORMANCE")
    lines.append(f"  total gross discharge revenue : ${perf['total_discharge_revenue']:,.4f}")
    lines.append(f"  total charging cost           : ${perf['total_charge_cost']:,.4f}")
    lines.append(f"  total net revenue             : ${perf['total_net_revenue']:,.4f}")
    lines.append(f"  total energy charged (grid)   : {perf['total_energy_charged_kwh']:,.4f} kWh")
    lines.append(f"  total energy discharged (grid): {perf['total_energy_discharged_kwh']:,.4f} kWh")
    lines.append(f"  battery throughput charged    : {perf['total_battery_charge_kwh']:,.4f} kWh")
    lines.append(f"  battery throughput discharged : {perf['total_battery_throughput_kwh']:,.4f} kWh")
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
    if perf["n_simultaneous_intervals"]:
        lines.append(f"  simultaneous-flow intervals   : {perf['n_simultaneous_intervals']} (LP relaxation only)")

    lines.append("")
    lines.append("MONTHLY GRID ENERGY USAGE (for retailer access-tier selection)")
    lines.append(f"  total energy imported (grid)  : {perf['total_energy_charged_kwh']:,.2f} kWh")
    lines.append(f"  total energy exported (grid)  : {perf['total_energy_discharged_kwh']:,.2f} kWh")
    lines.append(f"  net grid energy balance       : {perf['total_energy_charged_kwh'] - perf['total_energy_discharged_kwh']:,.2f} kWh")
    lines.append(f"  average daily import          : {perf['avg_daily_import_kwh']:,.2f} kWh")
    lines.append(f"  highest daily import          : {perf['max_daily_import_kwh']:,.2f} kWh")

    lines.append("")
    lines.append("RETAIL SUBSCRIPTION (Amber Electric - break-even reference)")
    lines.append(f"  grid import (period)          : {perf['total_energy_charged_kwh']:,.2f} kWh "
                 f"over {perf['amber_period_days']:,.2f} days")
    lines.append(f"  annualised usage (estimate)   : {perf['amber_annual_usage_kwh']:,.0f} kWh/yr")
    lines.append(f"  usage tier                    : {perf['amber_tier_label']} "
                 f"-> ${perf['amber_monthly_fee']:,.2f}/month")
    lines.append(f"  subscription cost (period)    : ${perf['amber_period_cost']:,.2f} "
                 f"({perf['amber_billing_months']} monthly fee"
                 f"{'s' if perf['amber_billing_months'] != 1 else ''})")
    lines.append("  (overlay only: not deducted from the revenue totals; annualising")
    lines.append("   the modelled period is an estimate, not a billed amount)")

    lines.append("")
    lines.append("DAILY PERFORMANCE")
    lines.append(f"  days in period                : {perf['n_days']}")
    lines.append(f"  intervals per full day        : {perf['intervals_per_day']}")
    lines.append(f"  partial days                  : {perf['n_partial_days']}")
    lines.append(f"  average daily revenue         : ${perf['avg_daily_revenue']:,.4f}")
    lines.append(f"  median daily revenue          : ${perf['median_daily_revenue']:,.4f}")
    lines.append(f"  highest day                   : {perf['best_day_date']}  ${perf['best_day_revenue']:,.4f}")
    lines.append(f"  lowest day                    : {perf['worst_day_date']}  ${perf['worst_day_revenue']:,.4f}")
    lines.append(f"  max rolling 24-hour revenue   : ${perf['max_rolling_24h_revenue']:,.4f}")
    lines.append(f"  min rolling 24-hour revenue   : ${perf['min_rolling_24h_revenue']:,.4f}")
    lines.append(f"  (day basis = {config.DAY_BASIS}; rolling 24h = {perf['intervals_per_day']} consecutive intervals)")

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
    lines.append(f"  simultaneous intervals        : {diag['simultaneous_intervals']}")
    lines.append(f"  max power-bound violation     : {diag['max_power_bound_violation_kw']:.3e} kW")
    lines.append(f"  missing intervals             : {data_report.n_missing_intervals}")
    lines.append(f"  optimality proven (gap<=target): {diag['globally_optimal']}")
    if diag.get("limit_reached"):
        lines.append("  NOTE: solver stopped at its limit; the schedule is feasible and")
        lines.append("        within the reported gap of the best bound, but NOT proven optimal.")
    lines.append(f"  MIP gap (HiGHS report)        : {_fmt_optional(diag.get('mip_gap'))}")
    lines.append(f"  MIP dual bound (objective)    : {_fmt_optional(diag.get('mip_dual_bound'))}")
    lines.append(f"  branch-and-bound nodes        : {diag.get('mip_nodes') if diag.get('mip_nodes') is not None else 'n/a'}")
    lines.append(f"  constraint violations         : {len(validation.violations)}")
    lines.append(f"  independent validation        : {'PASSED' if validation.passed else 'FAILED'}")

    # ---- optimality certificate --------------------------------------------
    lp_bound = diag.get("lp_bound_revenue")
    lines.append("")
    lines.append("OPTIMALITY CERTIFICATE")
    lines.append(f"  MILP objective (this run)     : ${-diag['objective_value']:,.6f}")
    if lp_bound is not None:
        gap_raw = float(lp_bound) - float(-diag["objective_value"])
        tol = max(BOUND_TOL_USD, 1e-9 * abs(lp_bound))
        gap = 0.0 if -tol <= gap_raw < 0.0 else gap_raw
        pct = 100.0 * gap / abs(lp_bound) if lp_bound else 0.0
        lines.append(f"  LP relaxation upper bound     : ${float(lp_bound):,.6f}")
        lines.append(f"  MILP gap vs LP bound          : ${gap:,.6f}  ({pct:.6f}%)")
        if gap_raw < -tol:
            lines.append("  bound check                   : FAILED (MILP exceeds LP bound!)")
        elif abs(gap_raw) <= tol:
            lines.append("  bound check                   : PASSED (MILP = LP bound to numerical tolerance)")
        else:
            lines.append("  bound check                   : PASSED (MILP <= LP bound)")
    else:
        lines.append("  LP relaxation upper bound     : not computed (COMPUTE_LP_BOUND=False)")
    for note in validation.notes:
        lines.append(f"  note: {note}")
    lines.append(sep)
    return "\n".join(lines)


def save_results(frame: pd.DataFrame, path: Path) -> Path:
    """Write the interval-level schedule to CSV for independent inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


__all__ = ["compute_performance", "format_summary", "save_results"]
