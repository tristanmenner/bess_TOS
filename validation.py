"""Independent validation of a TOS schedule.

Nothing here trusts the optimiser. Every physical and accounting quantity is
recomputed from the raw power schedule and compared against the values the
optimiser reported. If a constraint is violated the caller gets a hard error,
not a warning.

In MILP mode the schedule must also be physically realisable (no simultaneous
charge/discharge).  In LP-relaxation mode that constraint is deliberately
relaxed, so the count of simultaneous intervals is reported instead and the
schedule must be read as an upper bound, not as a dispatch plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import POWER_TOL_KW, REVENUE_TOL_USD, SOC_TOL_KWH, Config


class ConstraintViolationError(RuntimeError):
    """Raised when a validated schedule breaches a configured constraint."""


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"validation passed      : {self.passed}"]
        for name, ok in self.checks.items():
            lines.append(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        for name, value in self.details.items():
            lines.append(f"  {name}: {value:.6g}")
        lines.extend(f"  note: {n}" for n in self.notes)
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
    grid_import = config.grid_import_limit_kw
    grid_export = config.grid_export_limit_kw
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

    check("charge_power_non_negative", bool((c >= -POWER_TOL_KW).all()))
    check("discharge_power_non_negative", bool((d >= -POWER_TOL_KW).all()))
    check("charge_power_within_limit", bool((c <= p_charge + POWER_TOL_KW).all()))
    check("discharge_power_within_limit", bool((d <= p_discharge + POWER_TOL_KW).all()))
    # AC connection limits apply on top of the battery/converter limits.
    check("charge_power_within_grid_import_limit", bool((c <= grid_import + POWER_TOL_KW).all()))
    check("discharge_power_within_grid_export_limit", bool((d <= grid_export + POWER_TOL_KW).all()))
    check("soc_within_limits", bool((soc >= e_min - SOC_TOL_KWH).all() and (soc <= e_max + SOC_TOL_KWH).all()))

    # battery-side current never exceeds the C-rate / cable limit
    battery_charge_power = eta_c * c
    battery_discharge_power = d / eta_d
    check(
        "battery_side_charge_power_within_rating",
        bool((battery_charge_power <= config.dc_charge_power_kw + POWER_TOL_KW).all()),
    )
    check(
        "battery_side_discharge_power_within_rating",
        bool((battery_discharge_power <= config.dc_discharge_power_kw + POWER_TOL_KW).all()),
    )

    n_simultaneous = int(np.sum((c > POWER_TOL_KW) & (d > POWER_TOL_KW)))
    report.details["simultaneous_intervals"] = float(n_simultaneous)
    if config.ENFORCE_NO_SIMULTANEOUS:
        check("no_simultaneous_charge_discharge", n_simultaneous == 0)
    else:
        # LP relaxation: simultaneity is the known non-physical loophole; it is
        # reported, not treated as a validation failure.
        report.notes.append(
            "LP relaxation: simultaneous charge/discharge allowed; the schedule is "
            "an UPPER BOUND, not a physically realisable dispatch plan."
        )

    max_soc_err = float(np.max(np.abs(soc_rec_end - reported_soc_end)))
    report.details["soc_reconstruction_max_error_kwh"] = max_soc_err
    check("soc_consistent_with_schedule", max_soc_err <= max(SOC_TOL_KWH, 1e-8 * config.BATTERY_CAPACITY_KWH))

    rev_err = abs(float(frame["interval_revenue"].sum()) - total_revenue)
    cum_err = abs(float(frame["cumulative_revenue"].iloc[-1]) - total_revenue)
    report.details["revenue_reconstruction_error"] = max(rev_err, cum_err)
    check("revenue_consistent", max(rev_err, cum_err) <= REVENUE_TOL_USD)

    # terminal SOC
    final_soc = float(soc[-1])
    report.details["terminal_soc_kwh"] = final_soc
    if config.TERMINAL_SOC_MODE == "equal_initial":
        check("terminal_soc_equals_initial", abs(final_soc - e_init) <= SOC_TOL_KWH)
    else:
        check("terminal_soc_within_limits", e_min - SOC_TOL_KWH <= final_soc <= e_max + SOC_TOL_KWH)

    # energy accounting
    grid_charge = float((c * dt).sum())
    grid_discharge = float((d * dt).sum())
    report.details["grid_import_limit_kw"] = grid_import
    report.details["grid_export_limit_kw"] = grid_export
    report.details["grid_energy_charged_kwh"] = grid_charge
    report.details["grid_energy_discharged_kwh"] = grid_discharge
    report.details["total_revenue"] = total_revenue
    report.details["battery_throughput_charged_kwh"] = float(battery_charge_power.sum() * dt)
    report.details["battery_throughput_discharged_kwh"] = float(battery_discharge_power.sum() * dt)

    # Mode 1 daily cycle cap (basis per Config.MODE1_CYCLE_BASIS)
    if config.MODE == 1:
        counted = (
            config.mode1_cycle_coeff_c * c + config.mode1_cycle_coeff_d * d
        )
        daily = pd.DataFrame({"day": frame["day"].to_numpy(), "counted": counted})
        efc = daily.groupby("day")["counted"].sum() / config.usable_energy_kwh
        worst = float(efc.max())
        report.details["max_daily_efc"] = worst
        report.notes.append(
            f"Mode 1 cycle basis: {config.MODE1_CYCLE_BASIS} "
            f"({config.mode1_cycle_basis_label}); day basis: {config.DAY_BASIS}"
        )
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
