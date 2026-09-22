"""Independent validation of a TOS schedule.

Nothing here trusts the optimiser. Every physical and accounting quantity is
recomputed from the raw power schedule and compared against the values the
optimiser reported. If a constraint is violated the caller gets a hard error,
not a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import Config

SOC_TOL = 1e-6  # kWh
POWER_TOL = 1e-6  # kW
REVENUE_TOL = 1e-6  # $


class ConstraintViolationError(RuntimeError):
    """Raised when a validated schedule breaches a configured constraint."""


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"validation passed      : {self.passed}"]
        for name, ok in self.checks.items():
            lines.append(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        for name, value in self.details.items():
            lines.append(f"  {name}: {value:.6g}")
        lines.extend(f"  VIOLATION: {v}" for v in self.violations)
        return lines


def validate_schedule(config: Config, frame: pd.DataFrame) -> ValidationReport:
    """Recompute and verify every constraint and revenue figure.

    Raises
    ------
    ConstraintViolationError
        If any constraint is breached beyond numerical tolerance.
    """
    dt = config.interval_hours
    eta_c = config.charge_efficiency
    eta_d = config.discharge_efficiency
    p_charge = config.max_charge_power_kw
    p_discharge = config.max_discharge_power_kw
    e_min = config.min_energy_kwh
    e_max = config.max_energy_kwh
    e_init = config.initial_energy_kwh

    c = frame["charge_power_kw"].to_numpy(dtype=float)
    d = frame["discharge_power_kw"].to_numpy(dtype=float)
    price = frame["price"].to_numpy(dtype=float)
    reported_soc_end = frame["soc_kwh"].to_numpy(dtype=float)

    report = ValidationReport(passed=True)

    # ---- recompute SOC trajectory from the schedule ------------------------
    soc = np.empty(len(c) + 1)
    soc[0] = e_init
    for t in range(len(c)):
        soc[t + 1] = soc[t] + eta_c * dt * c[t] - (dt / eta_d) * d[t]
    soc_rec_end = soc[1:]

    # ---- recompute revenue -------------------------------------------------
    charge_cost = c * dt * price
    discharge_revenue = d * dt * price
    interval_revenue = discharge_revenue - charge_cost
    total_revenue = float(interval_revenue.sum())

    # ---- checks ------------------------------------------------------------
    def check(name: str, ok: bool) -> None:
        report.checks[name] = bool(ok)
        if not ok:
            report.passed = False

    check("charge_power_non_negative", bool((c >= -POWER_TOL).all()))
    check("discharge_power_non_negative", bool((d >= -POWER_TOL).all()))
    check("charge_power_within_limit", bool((c <= p_charge + POWER_TOL).all()))
    check("discharge_power_within_limit", bool((d <= p_discharge + POWER_TOL).all()))
    check("soc_within_limits", bool((soc >= e_min - SOC_TOL).all() and (soc <= e_max + SOC_TOL).all()))
    check("no_simultaneous_charge_discharge", bool(not ((c > POWER_TOL) & (d > POWER_TOL)).any()))

    max_soc_err = float(np.max(np.abs(soc_rec_end - reported_soc_end)))
    report.details["soc_reconstruction_max_error_kwh"] = max_soc_err
    check("soc_consistent_with_schedule", max_soc_err <= max(SOC_TOL, 1e-8 * config.BATTERY_CAPACITY_KWH))

    rev_err = abs(float(frame["interval_revenue"].sum()) - total_revenue)
    cum_err = abs(float(frame["cumulative_revenue"].iloc[-1]) - total_revenue)
    report.details["revenue_reconstruction_error"] = max(rev_err, cum_err)
    check("revenue_consistent", max(rev_err, cum_err) <= max(REVENUE_TOL, 1e-6))

    # terminal SOC
    final_soc = float(soc[-1])
    report.details["terminal_soc_kwh"] = final_soc
    if config.TERMINAL_SOC_MODE == "equal_initial":
        check("terminal_soc_equals_initial", abs(final_soc - e_init) <= max(SOC_TOL, 1e-6))
    else:
        check("terminal_soc_within_limits", e_min - SOC_TOL <= final_soc <= e_max + SOC_TOL)

    # energy accounting
    battery_discharge = d * dt / eta_d
    grid_charge = float((c * dt).sum())
    grid_discharge = float((d * dt).sum())
    report.details["grid_energy_charged_kwh"] = grid_charge
    report.details["grid_energy_discharged_kwh"] = grid_discharge
    report.details["total_revenue"] = total_revenue

    # Mode 1 daily EFC cap
    if config.MODE == 1:
        daily = pd.DataFrame({"day": frame["day"].to_numpy(), "batt_dis": battery_discharge})
        efc = daily.groupby("day")["batt_dis"].sum() / config.usable_energy_kwh
        worst = float(efc.max())
        report.details["max_daily_efc"] = worst
        check(
            "daily_cycle_cap_respected",
            worst <= config.CYCLES_PER_DAY + 1e-6,
        )

    if not report.passed:
        failed = [k for k, v in report.checks.items() if not v]
        raise ConstraintViolationError(
            "Schedule validation failed: " + ", ".join(failed)
        )
    return report


__all__ = ["ConstraintViolationError", "ValidationReport", "validate_schedule"]
