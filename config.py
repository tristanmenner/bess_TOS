"""Central configuration for the BESS Theoretically Optimal Strategy (TOS).

Everything a user is expected to change lives here. The optimisation
implementation reads a :class:`Config` instance and never hard-codes any of
these settings.

Conventions documented here (and used throughout the project)
-------------------------------------------------------------
* ``REGION``            NEM region identifier, one of NSW1 / QLD1 / SA1 / TAS1 / VIC1.
* ``START_DATE`` / ``END_DATE``
                        Inclusive calendar dates, interpreted in NEM time.
* ``MODE``              1 = fixed daily cycle count, 2 = unconstrained timing.
* ``CYCLES_PER_DAY``    Equivalent Full Cycles (EFC) cap per calendar day (Mode 1).
* Efficiency            ``ROUND_TRIP_EFFICIENCY`` is split symmetrically:
                        ``charge_efficiency = discharge_efficiency = sqrt(RTE)``.
                        Independently configurable overrides are supported.
* Power                 Charge/discharge power is derived from current and
                        voltage: ``I = C-rate x NOMINAL_CAPACITY_AH`` (A), then
                        ``P = I x NOMINAL_VOLTAGE_V / 1000`` (kW). ``MAX_CURRENT_A``
                        is a hard cable-size limit; the effective current is
                        ``min(C-rate current, MAX_CURRENT_A)``.
* Time                  NEM settlement timestamps are interval-ENDING and
                        expressed in NEM time (AEST, UTC+10, *no* daylight
                        saving). All time handling is naive/UTC+10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


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
    BATTERY_CAPACITY_KWH: float = 29.1 #119.0
    NOMINAL_VOLTAGE_V: float = 48.5 #743.6
    NOMINAL_CAPACITY_AH: float = 600 #160.0
    MIN_SOC_PERCENT: float = 1.0
    MAX_SOC_PERCENT: float = 100.0
    INITIAL_SOC_PERCENT: float = 50.0

    # C-rates are defined against the nominal Ah capacity (raw current) and
    # converted to power using the nominal voltage.
    MAX_CHARGE_C_RATE: float = 0.5
    MAX_DISCHARGE_C_RATE: float = 1.0

    # Hard BESS current limit set by the cable size (A). Applies to both charge
    # and discharge and can only ever tighten the C-rate limit:
    #     I_effective = min(C-rate x Ah, MAX_CURRENT_A)
    MAX_CURRENT_A: float = 100.0

    ROUND_TRIP_EFFICIENCY: float = 0.90
    # Optional independent overrides; leave as None to use the symmetric split.
    CHARGE_EFFICIENCY: float | None = None
    DISCHARGE_EFFICIENCY: float | None = None

    # ------------------------------------------------------------------ modes
    MODE: int = 2
    CYCLES_PER_DAY: float = 1.0

    # "equal_initial" forces final SOC back to initial SOC (sound benchmark).
    # "free" leaves the terminal SOC unconstrained within the SOC limits.
    TERMINAL_SOC_MODE: str = "equal_initial"

    # ------------------------------------------------------------------ model
    INTERVAL_MINUTES: float = 5.0
    PRICE_COLUMN: str = "RRP"
    PRICE_UNITS: str = "$/MWh"

    # Simultaneous charge/discharge must be blocked: without binaries the LP can
    # exploit negative prices non-physically (see optimizer.py). True selects the
    # MILP (correct); False selects the LP relaxation (optimistic upper bound).
    ENFORCE_NO_SIMULTANEOUS: bool = True

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

    @property
    def max_charge_power_kw(self) -> float:
        return self.max_charge_current_a * self.NOMINAL_VOLTAGE_V / 1000.0

    @property
    def max_discharge_power_kw(self) -> float:
        return self.max_discharge_current_a * self.NOMINAL_VOLTAGE_V / 1000.0

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
        if not 0.0 < self.MIN_SOC_PERCENT < self.MAX_SOC_PERCENT <= 100.0:
            raise ValueError("Require 0 < MIN_SOC_PERCENT < MAX_SOC_PERCENT <= 100")
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


# Default configuration used by main.py. Edit the values above (or replace this
# object) to change a run; the implementation is fully parameter driven.
CONFIG = Config()

__all__ = ["Config", "CONFIG", "PROJECT_DIR"]
