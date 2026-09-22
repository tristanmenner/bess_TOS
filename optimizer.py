"""Mixed-integer linear-programming Theoretically Optimal Strategy (TOS).

Mathematical formulation
------------------------
Index intervals ``t = 0 .. T-1``. Decision variables, per interval:

* ``c[t] >= 0``  grid/meter-side charging power (kW)
* ``d[t] >= 0``  grid/meter-side discharging power (kW)
* ``y[t] in {0,1}``  operating-mode binary: 1 = charging, 0 = discharging/idle
* ``E[k] >= 0``  battery-side stored energy (kWh) at interval boundary ``k``

Energy conventions (documented once, used consistently):

* ``charge_power`` / ``discharge_power`` are **grid-side** quantities. The
  settlement meter sees ``c[t]`` imported and ``d[t]`` exported.
* ``E`` is **battery-side** stored energy.
* ``charge_efficiency`` (eta_c) applies grid -> battery; ``discharge_efficiency``
  (eta_d) applies battery -> grid. With only a round-trip figure,
  ``eta_c = eta_d = sqrt(RTE)``.

Dynamics (battery-side energy balance, dimensionally kWh):

    E[t+1] = E[t] + eta_c * dt * c[t] - (dt / eta_d) * d[t]

Objective (maximise net meter-side revenue):

    max  sum_t dt * price_kwh[t] * (d[t] - c[t])

Constraints:

* ``0 <= c[t] <= P_charge_max``, ``0 <= d[t] <= P_discharge_max``
* ``E_min <= E[k] <= E_max``
* ``E[0] = initial energy``
* terminal SOC: ``E[T] = E[0]`` (equal_initial) or free within limits
* no simultaneous charge/discharge, enforced by binaries:
  ``c[t] <= P_charge_max * y[t]`` and ``d[t] <= P_discharge_max * (1 - y[t])``
* Mode 1 only: for each calendar day ``D``,
  ``sum_{t in D} (dt / eta_d) * d[t] <= CYCLES_PER_DAY * usable_energy``
  (battery-side discharge throughput, i.e. Equivalent Full Cycles).

Why binaries are genuinely required (and an LP is NOT sufficient)
-----------------------------------------------------------------
If simultaneous charge and discharge were allowed, the LP could exploit a
non-physical loophole at **negative** prices. Take ``c>0`` and ``d>0`` and
reduce both while preserving the SOC change, i.e. ``delta_d = eta_c*eta_d*delta_c``:

    dE       = eta_c*dt*c - (dt/eta_d)*d     is unchanged
    dRevenue = dt*price*(d - c)              changes by dt*price*delta_c*(1 - RTE)

When ``price > 0`` the change is negative, so simultaneity is unprofitable.
But when ``price < 0`` the change is *positive*: the LP is paid to route energy
through the round-trip losses, importing power while holding SOC constant.
That is not physically realisable through a single AC port. The binaries remove
it. (The LP therefore supplies an optimistic upper bound; the MILP is the
physically correct benchmark.)

Solver: HiGHS via ``scipy.optimize.milp``. HiGHS certifies proven global
optimality for the MILP (subject to MIP gap), which is reported per run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from config import Config

SOLVER_LP = "scipy.optimize.linprog / HiGHS (LP relaxation, upper bound)"
SOLVER_MILP = "scipy.optimize.milp / HiGHS (MILP, no-simultaneity enforced)"
POWER_TOL = 1e-7  # kW; anything below this is treated as zero flow


@dataclass
class SolveResult:
    """Raw optimiser output for a single run."""

    charge_power_kw: np.ndarray
    discharge_power_kw: np.ndarray
    soc_start_kwh: np.ndarray
    soc_end_kwh: np.ndarray
    diagnostics: dict


@dataclass
class _Model:
    n: int
    T: int
    obj: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    A_ub: sparse.csr_matrix | None
    b_ub: np.ndarray | None
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray | None
    c0: int
    d0: int
    e0: int


def _build_daily_groups(day_index: np.ndarray) -> list[np.ndarray]:
    """Return, for each distinct day id, the interval indices belonging to it."""
    order = np.argsort(day_index, kind="stable")
    sorted_days = day_index[order]
    boundaries = np.flatnonzero(np.diff(sorted_days)) + 1
    return np.split(order, boundaries)


def _build_model(config: Config, price_kwh: np.ndarray, day_index: np.ndarray) -> _Model:
    """Assemble the (MI)LP matrices. Shared by the LP and MILP paths."""
    price_kwh = np.asarray(price_kwh, dtype=float)
    day_index = np.asarray(day_index)
    T = price_kwh.size
    if T == 0:
        raise ValueError("Empty price series")
    if day_index.size != T:
        raise ValueError("day_index must align with price_kwh")

    dt = config.interval_hours
    eta_c = config.charge_efficiency
    eta_d = config.discharge_efficiency
    p_charge = config.max_charge_power_kw
    p_discharge = config.max_discharge_power_kw
    e_min = config.min_energy_kwh
    e_max = config.max_energy_kwh
    e_init = config.initial_energy_kwh

    use_binaries = bool(config.ENFORCE_NO_SIMULTANEOUS)
    c0, d0, e0 = 0, T, 2 * T
    y0 = 3 * T + 1
    n = (4 * T + 1) if use_binaries else (3 * T + 1)

    # ---- objective: minimise sum dt*price*(c - d) -------------------------
    obj = np.zeros(n)
    obj[c0 : c0 + T] = dt * price_kwh
    obj[d0 : d0 + T] = -dt * price_kwh

    # ---- equality: SOC dynamics (T rows) ----------------------------------
    idx = np.arange(T)
    A_eq = sparse.lil_matrix((T, n))
    A_eq[idx, e0 + idx + 1] = 1.0
    A_eq[idx, e0 + idx] = -1.0
    A_eq[idx, c0 + idx] = -eta_c * dt
    A_eq[idx, d0 + idx] = dt / eta_d
    A_eq = A_eq.tocsr()
    b_eq = np.zeros(T)

    # ---- inequalities ------------------------------------------------------
    ub_rows: list[sparse.csr_matrix] = []
    ub_rhs: list[np.ndarray] = []
    if config.MODE == 1:
        groups = _build_daily_groups(day_index)
        A_cyc = sparse.lil_matrix((len(groups), n))
        for r, idx_day in enumerate(groups):
            A_cyc[r, d0 + idx_day] = dt / eta_d  # battery-side discharge kWh
        ub_rows.append(A_cyc.tocsr())
        ub_rhs.append(np.full(len(groups), config.CYCLES_PER_DAY * config.usable_energy_kwh))
    if use_binaries:
        A_bin = sparse.lil_matrix((2 * T, n))
        A_bin[idx, c0 + idx] = 1.0
        A_bin[idx, y0 + idx] = -p_charge
        A_bin[T + idx, d0 + idx] = 1.0
        A_bin[T + idx, y0 + idx] = p_discharge
        ub_rows.append(A_bin.tocsr())
        ub_rhs.append(np.concatenate([np.zeros(T), np.full(T, p_discharge)]))
    A_ub = sparse.vstack(ub_rows).tocsr() if ub_rows else None
    b_ub = np.concatenate(ub_rhs) if ub_rhs else None

    # ---- bounds -----------------------------------------------------------
    lb = np.empty(n)
    ub = np.empty(n)
    lb[c0 : c0 + T] = 0.0
    ub[c0 : c0 + T] = p_charge
    lb[d0 : d0 + T] = 0.0
    ub[d0 : d0 + T] = p_discharge
    lb[e0] = ub[e0] = e_init
    lb[e0 + 1 : e0 + T] = e_min
    ub[e0 + 1 : e0 + T] = e_max
    if config.TERMINAL_SOC_MODE == "equal_initial":
        lb[e0 + T] = ub[e0 + T] = e_init
    else:
        lb[e0 + T] = e_min
        ub[e0 + T] = e_max
    if use_binaries:
        lb[y0 : y0 + T] = 0.0
        ub[y0 : y0 + T] = 1.0

    integrality = None
    if use_binaries:
        integrality = np.zeros(n)
        integrality[y0 : y0 + T] = 1

    return _Model(
        n=n, T=T, obj=obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        lb=lb, ub=ub, integrality=integrality, c0=c0, d0=d0, e0=e0,
    )


def solve_schedule(config: Config, price_kwh: np.ndarray, day_index: np.ndarray) -> SolveResult:
    """Solve the TOS. Uses the MILP (default) or LP path per config."""
    model = _build_model(config, price_kwh, day_index)
    if config.ENFORCE_NO_SIMULTANEOUS:
        return _solve_milp(config, price_kwh, model)
    return _solve_lp(config, price_kwh, model)


def _extract(config: Config, price_kwh: np.ndarray, model: _Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    dt = config.interval_hours
    c = np.clip(x[model.c0 : model.c0 + model.T], 0.0, config.max_charge_power_kw)
    d = np.clip(x[model.d0 : model.d0 + model.T], 0.0, config.max_discharge_power_kw)
    e = x[model.e0 : model.e0 + model.T + 1]
    n_fixed = 0
    if not config.ENFORCE_NO_SIMULTANEOUS:
        c, d, n_fixed = canonicalise_schedule(config, c, d)
    return c, d, e, n_fixed


def _solve_milp(config: Config, price_kwh: np.ndarray, model: _Model) -> SolveResult:
    A = sparse.vstack([model.A_eq, model.A_ub]).tocsr() if model.A_ub is not None else model.A_eq
    lb = np.concatenate([model.b_eq, -np.inf * np.ones(model.A_ub.shape[0])]) if model.A_ub is not None else model.b_eq
    ub = np.concatenate([model.b_eq, model.b_ub]) if model.A_ub is not None else model.b_eq

    t0 = time.perf_counter()
    res = milp(
        c=model.obj,
        integrality=model.integrality,
        bounds=Bounds(model.lb, model.ub),
        constraints=LinearConstraint(A, lb, ub),
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"MILP failed (status={res.status}): {res.message}")

    c, d, e, n_fixed = _extract(config, price_kwh, model, res.x)
    diag = _diagnostics(config, SOLVER_MILP, res.status, res.message, res.fun, runtime, model, n_fixed, price_kwh, c, d)
    return SolveResult(c, d, e[:-1], e[1:], diag)


def _solve_lp(config: Config, price_kwh: np.ndarray, model: _Model) -> SolveResult:
    t0 = time.perf_counter()
    res = linprog(
        c=model.obj,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=list(zip(model.lb, model.ub)),
        method="highs",
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"LP failed (status={res.status}): {res.message}")

    c, d, e, n_fixed = _extract(config, price_kwh, model, res.x)
    diag = _diagnostics(config, SOLVER_LP, res.status, res.message, res.fun, runtime, model, n_fixed, price_kwh, c, d)
    return SolveResult(c, d, e[:-1], e[1:], diag)


def _diagnostics(
    config, solver, status, message, objective, runtime, model, n_fixed, price_kwh, c, d
) -> dict:
    dt = config.interval_hours
    total_revenue = float(np.sum(dt * price_kwh * (d - c)))
    return {
        "solver": solver,
        "solver_status": int(status),
        "solver_message": str(message),
        "globally_optimal": bool(status == 0),
        "objective_value": float(objective),
        "total_revenue": total_revenue,
        "runtime_seconds": runtime,
        "n_intervals": int(model.T),
        "n_variables": int(model.n),
        "n_equality_constraints": int(model.T),
        "n_inequality_constraints": 0 if model.A_ub is None else int(model.A_ub.shape[0]),
        "n_binary_variables": int(model.integrality.sum()) if model.integrality is not None else 0,
        "simultaneous_intervals_canonicalised": int(n_fixed),
        "mode": config.MODE,
        "terminal_soc_mode": config.TERMINAL_SOC_MODE,
        "no_simultaneous_enforced": bool(config.ENFORCE_NO_SIMULTANEOUS),
        "max_charge_power_kw": config.max_charge_power_kw,
        "max_discharge_power_kw": config.max_discharge_power_kw,
        "max_charge_current_a": config.max_charge_current_a,
        "max_discharge_current_a": config.max_discharge_current_a,
        "charge_current_limited_by_cable": bool(config.charge_current_limited_by_cable),
        "discharge_current_limited_by_cable": bool(config.discharge_current_limited_by_cable),
    }


def canonicalise_schedule(
    config: Config, c: np.ndarray, d: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove simultaneous charge/discharge (LP path only).

    Reduces ``c`` by ``delta`` and ``d`` by ``eta_c*eta_d*delta`` so the SOC
    change is unchanged. This may *reduce* revenue at negative prices (the LP
    loophole), which is why the MILP is the default and correct model.
    """
    eta_c = config.charge_efficiency
    eta_d = config.discharge_efficiency
    c = c.copy()
    d = d.copy()
    n_fixed = 0
    for t in np.flatnonzero((c > tol) & (d > tol)):
        delta = min(c[t], d[t] / (eta_c * eta_d))
        if delta <= 0:
            continue
        c[t] -= delta
        d[t] -= eta_c * eta_d * delta
        n_fixed += 1
    c[c < tol] = 0.0
    d[d < tol] = 0.0
    return c, d, n_fixed


def build_schedule_frame(
    config: Config, tidy_prices: pd.DataFrame, result: SolveResult
) -> pd.DataFrame:
    """Assemble the interval-level results DataFrame (README section 21)."""
    dt = config.interval_hours
    c = result.charge_power_kw
    d = result.discharge_power_kw
    eta_c = config.charge_efficiency
    eta_d = config.discharge_efficiency

    charge_energy = c * dt
    discharge_energy = d * dt
    battery_charge = eta_c * charge_energy
    battery_discharge = discharge_energy / eta_d

    price = tidy_prices["price_kwh"].to_numpy()
    charge_cost = charge_energy * price
    discharge_revenue = discharge_energy * price
    interval_revenue = discharge_revenue - charge_cost

    soc_end = result.soc_end_kwh
    soc_percent = soc_end / config.BATTERY_CAPACITY_KWH * 100.0
    state = np.where(c > POWER_TOL, "charge", np.where(d > POWER_TOL, "discharge", "idle"))
    efc = np.cumsum(battery_discharge) / config.usable_energy_kwh

    return pd.DataFrame(
        {
            "timestamp": tidy_prices["timestamp"].to_numpy(),
            "region": tidy_prices["region"].to_numpy(),
            "price_mwh": tidy_prices["price_mwh"].to_numpy(),
            "price": price,
            "charge_power_kw": c,
            "discharge_power_kw": d,
            "charge_energy_kwh": charge_energy,
            "discharge_energy_kwh": discharge_energy,
            "battery_charge_kwh": battery_charge,
            "battery_discharge_kwh": battery_discharge,
            "soc_start_kwh": result.soc_start_kwh,
            "soc_kwh": soc_end,
            "soc_percent": soc_percent,
            "charge_cost": charge_cost,
            "discharge_revenue": discharge_revenue,
            "interval_revenue": interval_revenue,
            "cumulative_revenue": np.cumsum(interval_revenue),
            "cumulative_charge_energy_kwh": np.cumsum(charge_energy),
            "cumulative_discharge_energy_kwh": np.cumsum(discharge_energy),
            "operating_state": state,
            "equivalent_full_cycles": efc,
            "day": tidy_prices["day"].to_numpy(),
        }
    )


__all__ = [
    "SOLVER_LP",
    "SOLVER_MILP",
    "SolveResult",
    "solve_schedule",
    "canonicalise_schedule",
    "build_schedule_frame",
]
