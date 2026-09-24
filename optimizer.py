"""Mixed-integer linear-programming Theoretically Optimal Strategy (TOS).

Mathematical formulation
------------------------
Index intervals ``t = 0 .. T-1``. Decision variables, per interval:

* ``c[t] >= 0``  meter-side charging power (kW)
* ``d[t] >= 0``  meter-side discharging power (kW)
* ``y[t] in {0,1}``  operating-mode binary: 1 = charging, 0 = discharging/idle
* ``E[k] >= 0``  battery-side stored energy (kWh) at interval boundary ``k``

Energy conventions (documented once, used consistently):

* ``charge_power`` / ``discharge_power`` are **meter-side** quantities. The
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

* ``0 <= c[t] <= P_charge_eff``, ``0 <= d[t] <= P_discharge_eff`` where the
  limits are the **meter-side** equivalents of the battery/DC-side C-rate and
  cable-current ratings, tightened by the AC grid connection:
  ``P_charge_max = P_dc_charge / eta_c`` and
  ``P_discharge_max = P_dc_discharge * eta_d`` keep the battery-side current
  within ``min(C-rate * Ah, MAX_CURRENT_A)`` in both directions;
  ``P_charge_eff = min(P_charge_max, grid_import_limit_kw)`` and
  ``P_discharge_eff = min(P_discharge_max, grid_export_limit_kw)`` then apply
  the connection capacity (phases x per-phase current x phase voltage) and any
  static ``MAX_IMPORT_KW`` / ``MAX_EXPORT_KW`` cap.  With no site load or
  generation modelled, these bounds are exactly the net meter-flow caps
  (charge = import, discharge = export).
* ``E_min <= E[k] <= E_max``
* ``E[0] = initial energy``
* terminal SOC: ``E[T] = E[0]`` (equal_initial) or free within limits
* no simultaneous charge/discharge, enforced by binaries:
  ``c[t] <= P_charge_max * y[t]`` and ``d[t] <= P_discharge_max * (1 - y[t])``
* Mode 1 only: for each day ``D``,
  ``sum_{t in D} (coeff_c * c[t] + coeff_d * d[t]) <= CYCLES_PER_DAY * usable_energy``
  where the coefficients implement ``Config.MODE1_CYCLE_BASIS`` (battery-side
  discharge by default, i.e. Equivalent Full Cycles) and the day definition is
  ``Config.DAY_BASIS`` (calendar day by default).

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
it.

Solver: HiGHS via ``scipy.optimize.milp``. ``Config.MIP_REL_GAP`` defaults to
1e-4 (the HiGHS default); the achieved gap and dual bound are always reported,
and the run only claims optimality when the achieved gap meets the target. Set
``MIP_REL_GAP = 0`` to prove exact optimality (tractable for small packs).
``solve_lp_bound`` returns the LP relaxation value, which is an independent
upper bound on the physical optimum (the MILP objective must not exceed it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from config import POWER_TOL_KW, Config

SOLVER_LP = "scipy.optimize.linprog / HiGHS (LP relaxation, upper bound)"
SOLVER_MILP = "scipy.optimize.milp / HiGHS (MILP, no-simultaneity enforced)"


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


def _build_model(
    config: Config,
    price_kwh: np.ndarray,
    day_index: np.ndarray,
    enforce_no_simultaneous: bool | None = None,
) -> _Model:
    """Assemble the (MI)LP matrices. Shared by the LP and MILP paths.

    ``enforce_no_simultaneous`` overrides ``Config.ENFORCE_NO_SIMULTANEOUS``;
    passing ``False`` builds the LP relaxation used as the optimality bound.
    """
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
    p_charge = config.effective_charge_power_kw
    p_discharge = config.effective_discharge_power_kw
    e_min = config.min_energy_kwh
    e_max = config.max_energy_kwh
    e_init = config.initial_energy_kwh

    use_binaries = bool(
        config.ENFORCE_NO_SIMULTANEOUS
        if enforce_no_simultaneous is None
        else enforce_no_simultaneous
    )
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
        coeff_c = config.mode1_cycle_coeff_c
        coeff_d = config.mode1_cycle_coeff_d
        for r, idx_day in enumerate(groups):
            if coeff_c:
                A_cyc[r, c0 + idx_day] = coeff_c
            if coeff_d:
                A_cyc[r, d0 + idx_day] = coeff_d
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


def solve_lp_bound(config: Config, price_kwh: np.ndarray, day_index: np.ndarray) -> float:
    """LP relaxation revenue bound (>= physical optimum), as an independent check."""
    model = _build_model(config, price_kwh, day_index, enforce_no_simultaneous=False)
    res = linprog(
        c=model.obj,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=list(zip(model.lb, model.ub)),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP relaxation failed (status={res.status}): {res.message}")
    return float(-res.fun)


def _extract(
    config: Config, model: _Model, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Slice the solver vector and record any bound violation before clipping."""
    c_raw = np.asarray(x[model.c0 : model.c0 + model.T], dtype=float)
    d_raw = np.asarray(x[model.d0 : model.d0 + model.T], dtype=float)
    p_charge = config.effective_charge_power_kw
    p_discharge = config.effective_discharge_power_kw
    violation = 0.0
    if c_raw.size:
        violation = max(
            violation,
            float(max(0.0, -c_raw.min())),
            float(max(0.0, (c_raw - p_charge).max())),
            float(max(0.0, -d_raw.min())),
            float(max(0.0, (d_raw - p_discharge).max())),
        )
    c = np.clip(c_raw, 0.0, p_charge)
    d = np.clip(d_raw, 0.0, p_discharge)
    e = np.asarray(x[model.e0 : model.e0 + model.T + 1], dtype=float)
    return c, d, e, violation


def _solve_milp(config: Config, price_kwh: np.ndarray, model: _Model) -> SolveResult:
    A = sparse.vstack([model.A_eq, model.A_ub]).tocsr() if model.A_ub is not None else model.A_eq
    lb = np.concatenate([model.b_eq, -np.inf * np.ones(model.A_ub.shape[0])]) if model.A_ub is not None else model.b_eq
    ub = np.concatenate([model.b_eq, model.b_ub]) if model.A_ub is not None else model.b_eq

    options: dict = {"mip_rel_gap": float(config.MIP_REL_GAP)}
    if config.MIP_TIME_LIMIT_S is not None:
        options["time_limit"] = float(config.MIP_TIME_LIMIT_S)

    t0 = time.perf_counter()
    res = milp(
        c=model.obj,
        integrality=model.integrality,
        bounds=Bounds(model.lb, model.ub),
        constraints=LinearConstraint(A, lb, ub),
        options=options,
    )
    runtime = time.perf_counter() - t0
    has_solution = getattr(res, "x", None) is not None
    if not res.success and not has_solution:
        raise RuntimeError(f"MILP failed (status={res.status}): {res.message}")
    limit_reached = not bool(res.success)

    c, d, e, violation = _extract(config, model, res.x)
    gap = getattr(res, "mip_gap", None)
    diag = _diagnostics(
        config, SOLVER_MILP, res.status, res.message, res.fun, runtime, model,
        price_kwh, c, d, violation,
        mip_gap=None if gap is None else float(gap),
        mip_dual_bound=getattr(res, "mip_dual_bound", None),
        mip_nodes=getattr(res, "mip_node_count", None),
        limit_reached=limit_reached,
    )
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

    c, d, e, violation = _extract(config, model, res.x)
    diag = _diagnostics(
        config, SOLVER_LP, res.status, res.message, res.fun, runtime, model,
        price_kwh, c, d, violation,
        mip_gap=None, mip_dual_bound=None, mip_nodes=None,
    )
    return SolveResult(c, d, e[:-1], e[1:], diag)


def _diagnostics(
    config, solver, status, message, objective, runtime, model, price_kwh, c, d,
    max_bound_violation_kw=0.0, mip_gap=None, mip_dual_bound=None, mip_nodes=None,
    limit_reached=False,
) -> dict:
    dt = config.interval_hours
    total_revenue = float(np.sum(dt * price_kwh * (d - c)))
    simultaneous = int(np.sum((c > POWER_TOL_KW) & (d > POWER_TOL_KW)))
    if solver == SOLVER_MILP:
        gap_tol = max(float(config.MIP_REL_GAP), 1e-9)
        proven = bool(status == 0 and (mip_gap is None or float(mip_gap) <= gap_tol * 1.000001))
    else:
        proven = False
    return {
        "solver": solver,
        "solver_status": int(status),
        "solver_message": str(message),
        "limit_reached": bool(limit_reached),
        "globally_optimal": proven,
        "objective_value": float(objective),
        "total_revenue": total_revenue,
        "runtime_seconds": runtime,
        "n_intervals": int(model.T),
        "n_variables": int(model.n),
        "n_equality_constraints": int(model.T),
        "n_inequality_constraints": 0 if model.A_ub is None else int(model.A_ub.shape[0]),
        "n_binary_variables": int(model.integrality.sum()) if model.integrality is not None else 0,
        "mip_rel_gap_requested": float(config.MIP_REL_GAP),
        "mip_gap": mip_gap,
        "mip_dual_bound": None if mip_dual_bound is None else float(mip_dual_bound),
        "mip_nodes": None if mip_nodes is None else int(mip_nodes),
        "lp_bound_revenue": None,
        "bound_gap_absolute": None,
        "simultaneous_intervals": simultaneous,
        "max_power_bound_violation_kw": float(max_bound_violation_kw),
        "mode": config.MODE,
        "mode1_cycle_basis": config.MODE1_CYCLE_BASIS,
        "mode1_cycle_basis_label": config.mode1_cycle_basis_label,
        "day_basis": config.DAY_BASIS,
        "terminal_soc_mode": config.TERMINAL_SOC_MODE,
        "no_simultaneous_enforced": bool(config.ENFORCE_NO_SIMULTANEOUS),
        "max_charge_power_kw": config.max_charge_power_kw,
        "max_discharge_power_kw": config.max_discharge_power_kw,
        "connection_power_kw": config.connection_power_kw,
        "phases": config.PHASES,
        "max_current_a_per_phase": config.MAX_CURRENT_A_PER_PHASE,
        "nominal_ac_voltage_per_phase_v": config.NOMINAL_AC_VOLTAGE_PER_PHASE_V,
        "max_export_kw": config.MAX_EXPORT_KW,
        "max_import_kw": config.MAX_IMPORT_KW,
        "grid_import_limit_kw": config.grid_import_limit_kw,
        "grid_export_limit_kw": config.grid_export_limit_kw,
        "effective_charge_power_kw": config.effective_charge_power_kw,
        "effective_discharge_power_kw": config.effective_discharge_power_kw,
        "charge_limited_by_grid": bool(config.charge_limited_by_grid),
        "discharge_limited_by_grid": bool(config.discharge_limited_by_grid),
        "dc_charge_power_kw": config.dc_charge_power_kw,
        "dc_discharge_power_kw": config.dc_discharge_power_kw,
        "max_charge_current_a": config.max_charge_current_a,
        "max_discharge_current_a": config.max_discharge_current_a,
        "charge_current_limited_by_cable": bool(config.charge_current_limited_by_cable),
        "discharge_current_limited_by_cable": bool(config.discharge_current_limited_by_cable),
    }


def canonicalise_schedule(
    config: Config, c: np.ndarray, d: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove simultaneous charge/discharge from a relaxation solution.

    Reduces ``c`` by ``delta`` and ``d`` by ``eta_c*eta_d*delta`` so the SOC
    change is unchanged.  This produces a *feasible but generally sub-optimal*
    physical schedule; it is NOT an upper bound and is therefore not applied to
    the LP path by default (see the README).  Kept as an explicit utility for
    users who want to post-process a relaxation solution.
    """
    eta_c = config.charge_efficiency
    eta_d = config.discharge_efficiency
    c = np.array(c, dtype=float, copy=True)
    d = np.array(d, dtype=float, copy=True)
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
    state = np.where(
        (c > POWER_TOL_KW) & (d > POWER_TOL_KW),
        "simultaneous",
        np.where(c > POWER_TOL_KW, "charge", np.where(d > POWER_TOL_KW, "discharge", "idle")),
    )
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
    "solve_lp_bound",
    "canonicalise_schedule",
    "build_schedule_frame",
]
