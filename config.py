"""Central configuration for the BESS Theoretically Optimal Strategy (TOS).

Everything a user is expected to change lives here. The optimisation
implementation reads a :class:`Config` instance and never hard-codes any of
these settings.

Conventions documented here (and used throughout the project)
-------------------------------------------------------------
* ``REGION``            NEM region identifier, one of NSW1 / QLD1 / SA1 / TAS1 / VIC1.
* ``START_DATE`` / ``END_DATE``
                        Inclusive calendar dates, interpreted in NEM time.
* ``MODE``              1 = fixed daily cycle cap, 2 = unconstrained timing.
* ``CYCLES_PER_DAY``    Equivalent Full Cycles (EFC) cap per day (Mode 1).
                        ``MODE1_CYCLE_BASIS`` selects which energy flow the cap
                        counts; ``DAY_BASIS`` selects the day definition.
* Efficiency            ``ROUND_TRIP_EFFICIENCY`` is split symmetrically:
                        ``charge_efficiency = discharge_efficiency = sqrt(RTE)``.
                        Independently configurable overrides are supported.
* Power                 Charge/discharge power is derived from current and
                        voltage: ``I = C-rate x NOMINAL_CAPACITY_AH`` (A), then
                        ``P_dc = I x NOMINAL_VOLTAGE_V / 1000`` (kW).
                        ``MAX_CURRENT_A`` is a hard cable-size limit; the
                        effective current is
                        ``min(C-rate current, MAX_CURRENT_A)``.

                        ``P_dc`` is a **battery/DC-side** rating (current times
                        DC voltage).  The optimiser constrains meter-side
                        flows, so the conversion through the converter
                        efficiency is applied explicitly:

                            c <= P_dc_charge / eta_c          (meter-side charge)
                            d <= P_dc_discharge * eta_d       (meter-side discharge)

                        ``max_charge_power_kw`` / ``max_discharge_power_kw``
                        return these meter-side limits; the DC ratings are
                        available as ``dc_charge_power_kw`` /
                        ``dc_discharge_power_kw``.
* Grid connection       AC connection capacity is
                        ``PHASES x MAX_CURRENT_A_PER_PHASE x
                        NOMINAL_AC_VOLTAGE_PER_PHASE_V``.  It is a hard site
                        limit on meter-side import and export; optional
                        ``MAX_EXPORT_KW`` / ``MAX_IMPORT_KW`` tighten it
                        (e.g. a static export limit).  The battery/converter
                        and grid limits both apply, so the effective limit is
                        the tighter of the two (``effective_charge_power_kw`` /
                        ``effective_discharge_power_kw``).
* Time                  NEM settlement timestamps are interval-ENDING and
                        expressed in NEM time (AEST, UTC+10, *no* daylight
                        saving). All time handling is naive/UTC+10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Shared numerical tolerances (kW / kWh / $).  The optimiser post-processing and
# the independent validation module import these so "zero" means the same thing
# everywhere.
# --------------------------------------------------------------------------- #
POWER_TOL_KW = 1e-6
SOC_TOL_KWH = 1e-6
REVENUE_TOL_USD = 1e-6
# Absolute slack allowed when comparing independently solved objectives (the LP
# bound and the MILP objective come from different algorithms/feasibility
# tolerances, so micro-dollar disagreements are numerical, not modelling, errors).
BOUND_TOL_USD = 1e-5

# Allowed values for the definitional switches (kept explicit on purpose: the
# "correct" Mode 1 depends on the specification being benchmarked).
MODE1_CYCLE_BASES = (
    "battery_discharge",  # battery-side kWh discharged  (EFC = out/usable)
    "battery_charge",     # battery-side kWh charged
    "grid_discharge",     # meter-side kWh exported
    "grid_charge",        # meter-side kWh imported
)
DAY_BASES = ("calendar", "nem_trading")


@dataclass(frozen=True)
class Config:
    """All user-tunable settings for a single TOS run."""

    # ------------------------------------------------------------------ market
    REGION: str = "VIC1"
    START_DATE: str = "2026-08-01"
    END_DATE: str = "2026-08-31"

    # ---------------------------------------------------------------- battery
    # Energy capacity, nominal pack voltage and nominal Ah are mutually
    # consistent: BATTERY_CAPACITY_KWH ~= NOMINAL_CAPACITY_AH * NOMINAL_VOLTAGE_V / 1000.
    BATTERY_CAPACITY_KWH: float = 119.0
    NOMINAL_VOLTAGE_V: float = 743.6
    NOMINAL_CAPACITY_AH: float = 160.0
    MIN_SOC_PERCENT: float = 0.0
    MAX_SOC_PERCENT: float = 90.0
    INITIAL_SOC_PERCENT: float = 50.0

    # C-rates are defined against the nominal Ah capacity (raw current) and
    # converted to power using the nominal voltage (battery/DC side).
    MAX_CHARGE_C_RATE: float = 0.5
    MAX_DISCHARGE_C_RATE: float = 1.0

    # Hard BESS current limit set by the cable size (A). Applies to both charge
    # and discharge and can only ever tighten the C-rate limit:
    #     I_effective = min(C-rate x Ah, MAX_CURRENT_A)
    MAX_CURRENT_A: float = 200.0

    ROUND_TRIP_EFFICIENCY: float = 0.90
    # Optional independent overrides; leave as None to use the symmetric split.
    CHARGE_EFFICIENCY: float | None = None
    DISCHARGE_EFFICIENCY: float | None = None

    # -------------------------------------------------------- grid connection
    # AC connection capacity = PHASES x MAX_CURRENT_A_PER_PHASE x
    # NOMINAL_AC_VOLTAGE_PER_PHASE_V.  This is a hard site limit on meter-side
    # import and export; the battery/DC ratings above still apply and the
    # tighter limit binds (see effective_charge_power_kw below).
    PHASES: int = 3
    NOMINAL_AC_VOLTAGE_PER_PHASE_V: float = 230.0
    MAX_CURRENT_A_PER_PHASE: float = 100.0
    # Optional static caps (e.g. a 4.5 kW static export limit).  None means the
    # connection capacity is the only grid limit.  Both are total site kW.
    MAX_EXPORT_KW: float | None = None
    MAX_IMPORT_KW: float | None = None

    # ------------------------------------------------------------------ modes
    MODE: int = 2
    CYCLES_PER_DAY: float = 2.0
    # Which flow the Mode 1 daily EFC cap counts.  The default is the battery-side
    # discharge throughput (1 EFC = usable_energy_kwh discharged at the battery).
    # This is a definitional choice - confirm it against the specification being
    # benchmarked before comparing runs that use different bases.
    MODE1_CYCLE_BASIS: str = "battery_discharge"
    # Day definition used both by the Mode 1 cap and by all daily reporting.
    # "calendar"    : midnight-to-midnight NEM time.
    # "nem_trading" : 04:00-to-04:00 NEM time (the AEMO trading day).  With this
    #                 basis the first and last days of a period are partial days.
    DAY_BASIS: str = "calendar"

    # "equal_initial" forces final SOC back to initial SOC (sound benchmark).
    # "free" leaves the terminal SOC unconstrained within the SOC limits.
    TERMINAL_SOC_MODE: str = "equal_initial"

    # ------------------------------------------------------------------ model
    INTERVAL_MINUTES: float = 5.0
    PRICE_COLUMN: str = "RRP"
    PRICE_UNITS: str = "$/MWh"

    # Simultaneous charge/discharge must be blocked: without binaries the LP can
    # exploit negative prices non-physically (see optimizer.py). True selects the
    # MILP (correct); False selects the LP relaxation (optimistic upper bound,
    # NOT a physically realisable schedule).
    ENFORCE_NO_SIMULTANEOUS: bool = True

    # MILP termination: the achieved gap is always reported (and the summary
    # only claims optimality when it meets this target).  1e-4 is the HiGHS
    # default and keeps large packs tractable; set 0.0 to prove exact optimality
    # (fine for small packs, potentially very slow for large ones).
    MIP_REL_GAP: float = 1e-4
    MIP_TIME_LIMIT_S: float | None = None

    # Solve the LP relaxation alongside the MILP and report it as an independent
    # optimality certificate (MILP objective <= LP bound).  Costs a few seconds.
    COMPUTE_LP_BOUND: bool = True

    # ------------------------------------------------------- data validation
    # A perfect-hindsight benchmark must cover the requested period.  With
    # REQUIRE_FULL_COVERAGE=True (default) a price series that does not cover
    # every expected interval of [START_DATE, END_DATE] raises instead of
    # silently pricing a shorter horizon.  Set False to accept partial data
    # (the gap is then reported as a warning and in the summary).
    REQUIRE_FULL_COVERAGE: bool = True

    # ------------------------------------------------------------------ paths
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_DIR / "data")
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_DIR / "outputs")
    PLOT_PATH: Path = field(default_factory=lambda: PROJECT_DIR / "generated_plot.png")

    # --------------------------------------------------------------- derived
    @property
    def interval_hours(self) -> float:
        """Duration of one settlement interval in hours (5 min -> 1/12 h)."""
        return self.INTERVAL_MINUTES / 60.0

    @property
    def usable_energy_kwh(self) -> float:
        """Energy available between the configured SOC limits (kWh)."""
        return (
            self.BATTERY_CAPACITY_KWH
            * (self.MAX_SOC_PERCENT - self.MIN_SOC_PERCENT)
            / 100.0
        )

    @property
    def min_energy_kwh(self) -> float:
        return self.BATTERY_CAPACITY_KWH * self.MIN_SOC_PERCENT / 100.0

    @property
    def max_energy_kwh(self) -> float:
        return self.BATTERY_CAPACITY_KWH * self.MAX_SOC_PERCENT / 100.0

    @property
    def initial_energy_kwh(self) -> float:
        return self.BATTERY_CAPACITY_KWH * self.INITIAL_SOC_PERCENT / 100.0

    @property
    def nominal_energy_from_ah_kwh(self) -> float:
        """Energy implied by the nominal Ah and voltage (consistency check)."""
        return self.NOMINAL_CAPACITY_AH * self.NOMINAL_VOLTAGE_V / 1000.0

    @property
    def c_rate_charge_current_a(self) -> float:
        """Charge current implied by the charge C-rate alone (A)."""
        return self.MAX_CHARGE_C_RATE * self.NOMINAL_CAPACITY_AH

    @property
    def c_rate_discharge_current_a(self) -> float:
        """Discharge current implied by the discharge C-rate alone (A)."""
        return self.MAX_DISCHARGE_C_RATE * self.NOMINAL_CAPACITY_AH

    @property
    def max_charge_current_a(self) -> float:
        """Effective charge current: C-rate limit tightened by the cable limit."""
        return min(self.c_rate_charge_current_a, self.MAX_CURRENT_A)

    @property
    def max_discharge_current_a(self) -> float:
        """Effective discharge current: C-rate limit tightened by the cable limit."""
        return min(self.c_rate_discharge_current_a, self.MAX_CURRENT_A)

    @property
    def charge_current_limited_by_cable(self) -> bool:
        """True when the cable current limit is the binding charge constraint."""
        return self.MAX_CURRENT_A < self.c_rate_charge_current_a

    @property
    def discharge_current_limited_by_cable(self) -> bool:
        """True when the cable current limit is the binding discharge constraint."""
        return self.MAX_CURRENT_A < self.c_rate_discharge_current_a

    # ------------------------------------------------- battery/DC-side ratings
    @property
    def dc_charge_power_kw(self) -> float:
        """Battery/DC-side charge power rating: I_effective x nominal voltage."""
        return self.max_charge_current_a * self.NOMINAL_VOLTAGE_V / 1000.0

    @property
    def dc_discharge_power_kw(self) -> float:
        """Battery/DC-side discharge power rating: I_effective x nominal voltage."""
        return self.max_discharge_current_a * self.NOMINAL_VOLTAGE_V / 1000.0

    # ----------------------------------------------------- meter-side limits
    @property
    def max_charge_power_kw(self) -> float:
        """Meter-side charge power limit (DC rating / eta_c).

        The C-rate/cable current limits the battery-side current; drawing the
        same battery-side power from the meter needs more meter-side power by
        the charge efficiency.
        """
        return self.dc_charge_power_kw / self.charge_efficiency

    @property
    def max_discharge_power_kw(self) -> float:
        """Meter-side discharge power limit (DC rating x eta_d).

        The battery-side discharge power is what the C-rate/cable current
        limits, and only eta_d of it reaches the meter.
        """
        return self.dc_discharge_power_kw * self.discharge_efficiency

    # ---------------------------------------------------- grid connection cap
    @property
    def connection_power_kw(self) -> float:
        """AC connection capacity from phases and per-phase current (kW)."""
        return (
            self.PHASES
            * self.MAX_CURRENT_A_PER_PHASE
            * self.NOMINAL_AC_VOLTAGE_PER_PHASE_V
            / 1000.0
        )

    @property
    def grid_import_limit_kw(self) -> float:
        """Site import cap: connection capacity tightened by MAX_IMPORT_KW."""
        limit = self.connection_power_kw
        if self.MAX_IMPORT_KW is not None:
            limit = min(limit, self.MAX_IMPORT_KW)
        return limit

    @property
    def grid_export_limit_kw(self) -> float:
        """Site export cap: connection capacity tightened by MAX_EXPORT_KW."""
        limit = self.connection_power_kw
        if self.MAX_EXPORT_KW is not None:
            limit = min(limit, self.MAX_EXPORT_KW)
        return limit

    @property
    def effective_charge_power_kw(self) -> float:
        """Meter-side charge limit: tighter of battery/converter and grid."""
        return min(self.max_charge_power_kw, self.grid_import_limit_kw)

    @property
    def effective_discharge_power_kw(self) -> float:
        """Meter-side discharge limit: tighter of battery/converter and grid."""
        return min(self.max_discharge_power_kw, self.grid_export_limit_kw)

    @property
    def charge_limited_by_grid(self) -> bool:
        """True when the AC connection/import cap is the binding charge limit."""
        return self.grid_import_limit_kw < self.max_charge_power_kw

    @property
    def discharge_limited_by_grid(self) -> bool:
        """True when the AC connection/export cap is the binding discharge limit."""
        return self.grid_export_limit_kw < self.max_discharge_power_kw

    # ------------------------------------------------------------ efficiency
    @property
    def charge_efficiency(self) -> float:
        if self.CHARGE_EFFICIENCY is not None:
            return float(self.CHARGE_EFFICIENCY)
        return math.sqrt(self.ROUND_TRIP_EFFICIENCY)

    @property
    def discharge_efficiency(self) -> float:
        if self.DISCHARGE_EFFICIENCY is not None:
            return float(self.DISCHARGE_EFFICIENCY)
        return math.sqrt(self.ROUND_TRIP_EFFICIENCY)

    # ---------------------------------------------------------- Mode 1 basis
    @property
    def mode1_cycle_coeff_c(self) -> float:
        """Counted cycle kWh per kW of grid-side charge power (Mode 1)."""
        if self.MODE1_CYCLE_BASIS == "battery_charge":
            return self.charge_efficiency * self.interval_hours
        if self.MODE1_CYCLE_BASIS == "grid_charge":
            return self.interval_hours
        return 0.0

    @property
    def mode1_cycle_coeff_d(self) -> float:
        """Counted cycle kWh per kW of grid-side discharge power (Mode 1)."""
        if self.MODE1_CYCLE_BASIS == "battery_discharge":
            return self.interval_hours / self.discharge_efficiency
        if self.MODE1_CYCLE_BASIS == "grid_discharge":
            return self.interval_hours
        return 0.0

    @property
    def mode1_cycle_basis_label(self) -> str:
        return {
            "battery_discharge": "battery-side discharge (1 EFC = usable kWh out of the battery)",
            "battery_charge": "battery-side charge (1 EFC = usable kWh into the battery)",
            "grid_discharge": "meter-side export",
            "grid_charge": "meter-side import",
        }[self.MODE1_CYCLE_BASIS]

    # ------------------------------------------------------------------ price
    @property
    def price_scale(self) -> float:
        """Multiplier converting the raw price column into $/kWh."""
        if self.PRICE_UNITS == "$/MWh":
            return 1.0 / 1000.0
        if self.PRICE_UNITS == "$/kWh":
            return 1.0
        raise ValueError(f"Unsupported PRICE_UNITS: {self.PRICE_UNITS!r}")

    def validate(self) -> None:
        """Fail fast on a nonsensical configuration."""
        if self.MODE not in (1, 2):
            raise ValueError(f"MODE must be 1 or 2, got {self.MODE!r}")
        if not 0.0 <= self.MIN_SOC_PERCENT < self.MAX_SOC_PERCENT <= 100.0:
            raise ValueError("Require 0 <= MIN_SOC_PERCENT < MAX_SOC_PERCENT <= 100")
        if not (self.MIN_SOC_PERCENT <= self.INITIAL_SOC_PERCENT <= self.MAX_SOC_PERCENT):
            raise ValueError("INITIAL_SOC_PERCENT must lie within the SOC limits")
        if self.ROUND_TRIP_EFFICIENCY <= 0.0 or self.ROUND_TRIP_EFFICIENCY > 1.0:
            raise ValueError("ROUND_TRIP_EFFICIENCY must be in (0, 1]")
        if self.charge_efficiency <= 0.0 or self.charge_efficiency > 1.0:
            raise ValueError("charge efficiency must be in (0, 1]")
        if self.discharge_efficiency <= 0.0 or self.discharge_efficiency > 1.0:
            raise ValueError("discharge efficiency must be in (0, 1]")
        if self.MAX_CHARGE_C_RATE <= 0.0 or self.MAX_DISCHARGE_C_RATE <= 0.0:
            raise ValueError("C-rates must be positive")
        if self.NOMINAL_VOLTAGE_V <= 0.0 or self.NOMINAL_CAPACITY_AH <= 0.0:
            raise ValueError("NOMINAL_VOLTAGE_V and NOMINAL_CAPACITY_AH must be positive")
        if self.MAX_CURRENT_A <= 0.0:
            raise ValueError("MAX_CURRENT_A must be positive")
        if self.PHASES not in (1, 3):
            raise ValueError(f"PHASES must be 1 or 3, got {self.PHASES!r}")
        if self.NOMINAL_AC_VOLTAGE_PER_PHASE_V <= 0.0:
            raise ValueError("NOMINAL_AC_VOLTAGE_PER_PHASE_V must be positive")
        if self.MAX_CURRENT_A_PER_PHASE <= 0.0:
            raise ValueError("MAX_CURRENT_A_PER_PHASE must be positive")
        if self.MAX_EXPORT_KW is not None and self.MAX_EXPORT_KW <= 0.0:
            raise ValueError("MAX_EXPORT_KW must be positive when set")
        if self.MAX_IMPORT_KW is not None and self.MAX_IMPORT_KW <= 0.0:
            raise ValueError("MAX_IMPORT_KW must be positive when set")
        implied = self.nominal_energy_from_ah_kwh
        tol = max(0.5, 0.01 * self.BATTERY_CAPACITY_KWH)
        if abs(implied - self.BATTERY_CAPACITY_KWH) > tol:
            raise ValueError(
                "Inconsistent battery spec: BATTERY_CAPACITY_KWH = "
                f"{self.BATTERY_CAPACITY_KWH} kWh, but NOMINAL_CAPACITY_AH * "
                f"NOMINAL_VOLTAGE_V / 1000 = {implied:.3f} kWh (tolerance {tol:.3f}). "
                "Adjust one of the three values so they agree."
            )
        if self.TERMINAL_SOC_MODE not in ("free", "equal_initial"):
            raise ValueError("TERMINAL_SOC_MODE must be 'free' or 'equal_initial'")
        if self.MODE == 1 and self.CYCLES_PER_DAY <= 0.0:
            raise ValueError("CYCLES_PER_DAY must be positive in Mode 1")
        if self.MODE1_CYCLE_BASIS not in MODE1_CYCLE_BASES:
            raise ValueError(
                f"MODE1_CYCLE_BASIS must be one of {MODE1_CYCLE_BASES}, "
                f"got {self.MODE1_CYCLE_BASIS!r}"
            )
        if self.DAY_BASIS not in DAY_BASES:
            raise ValueError(f"DAY_BASIS must be one of {DAY_BASES}, got {self.DAY_BASIS!r}")
        if self.INTERVAL_MINUTES <= 0.0:
            raise ValueError("INTERVAL_MINUTES must be positive")
        if self.MIP_REL_GAP < 0.0:
            raise ValueError("MIP_REL_GAP must be non-negative")
        if self.MIP_TIME_LIMIT_S is not None and self.MIP_TIME_LIMIT_S <= 0.0:
            raise ValueError("MIP_TIME_LIMIT_S must be positive when set")
        # Raises for an unsupported unit string (fail before the solver runs).
        self.price_scale


# Default configuration used by main.py. Edit the values above (or replace this
# object) to change a run; the implementation is fully parameter driven.
CONFIG = Config()

__all__ = [
    "Config",
    "CONFIG",
    "PROJECT_DIR",
    "POWER_TOL_KW",
    "SOC_TOL_KWH",
    "REVENUE_TOL_USD",
    "BOUND_TOL_USD",
    "MODE1_CYCLE_BASES",
    "DAY_BASES",
]
