"""Synthetic sanity tests for the TOS engine.

These run before any reliance on real NEM data and check that the optimiser and
its accounting behave correctly on problems with a known answer. Run with:

    python sanity_tests.py

Exits non-zero if any test fails.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from config import CONFIG, Config
from optimizer import build_schedule_frame, solve_schedule
from validation import validate_schedule

FAILURES: list[str] = []


def _series(prices_kwh: np.ndarray) -> pd.DataFrame:
    """Build a tidy price frame from an array of $/kWh prices at 5-min steps."""
    n = len(prices_kwh)
    interval_start = pd.date_range("2026-01-01 00:00", periods=n, freq="5min")
    days = interval_start.normalize()
    return pd.DataFrame(
        {
            "timestamp": interval_start + pd.Timedelta(minutes=5),
            "interval_start": interval_start,
            "region": "TEST",
            "price_mwh": prices_kwh * 1000.0,
            "price_kwh": prices_kwh,
            "day": days,
        }
    )


def _run(config: Config, prices_kwh: np.ndarray, label: str):
    tidy = _series(prices_kwh)
    day_index = pd.factorize(tidy["day"])[0]
    result = solve_schedule(config, prices_kwh, day_index)
    frame = build_schedule_frame(config, tidy, result)
    report = validate_schedule(config, frame)
    assert report.passed, f"{label}: validation failed"
    return frame, result


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def test_a_constant_price() -> None:
    print("Test A - constant price yields no artificial arbitrage")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2)
    frame, _ = _run(cfg, np.full(288, 0.08), "A")
    check(abs(frame["interval_revenue"].sum()) < 1e-6, "net revenue is zero")
    check((frame["charge_power_kw"].abs() < 1e-6).all(), "no charging occurs")
    check((frame["discharge_power_kw"].abs() < 1e-6).all(), "no discharging occurs")


def test_b_single_price_spike() -> None:
    print("Test B - cheap then expensive: charge low, discharge high")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2)
    prices = np.concatenate([np.full(144, 0.01), np.full(144, 0.50)])
    frame, _ = _run(cfg, prices, "B")
    low = frame["price"] < 0.1
    check((frame.loc[low, "discharge_power_kw"] < 1e-6).all(), "no discharge while cheap")
    check((frame.loc[~low, "charge_power_kw"] < 1e-6).all(), "no charge while expensive")
    check(frame["interval_revenue"].sum() > 0.0, "positive net revenue")
    check(frame["soc_percent"].max() > cfg.MAX_SOC_PERCENT - 1e-3, "SOC reaches max")
    check(abs(frame["soc_kwh"].iloc[-1] - cfg.initial_energy_kwh) < 1e-6, "terminal SOC = initial")


def test_c_negative_price() -> None:
    print("Test C - negative price creates a paid-to-charge incentive")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2)
    prices = np.concatenate([np.full(48, -0.05), np.full(240, 0.02)])
    frame, _ = _run(cfg, prices, "C")
    neg = frame["price"] < 0
    check(frame.loc[neg, "charge_power_kw"].sum() > 0, "charges during negative prices")
    check(frame["interval_revenue"].sum() > 0, "positive net revenue")
    check((frame.loc[neg, "charge_cost"] <= 0).all(), "negative cost (paid to charge)")


def test_d_soc_limits() -> None:
    print("Test D - SOC never breaches configured limits")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="free", MODE=2)
    prices = np.concatenate([np.full(96, -0.20), np.full(96, 0.40)])
    frame, _ = _run(cfg, prices, "D")
    check(frame["soc_percent"].min() >= cfg.MIN_SOC_PERCENT - 1e-6, "SOC >= min")
    check(frame["soc_percent"].max() <= cfg.MAX_SOC_PERCENT + 1e-6, "SOC <= max")


def test_e_power_limits() -> None:
    print("Test E - charge/discharge power respects C-rate limits")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="free", MODE=2)
    prices = np.concatenate([np.full(96, -0.20), np.full(96, 0.40)])
    frame, _ = _run(cfg, prices, "E")
    check(frame["charge_power_kw"].max() <= cfg.max_charge_power_kw + 1e-6, "charge power <= max")
    check(frame["discharge_power_kw"].max() <= cfg.max_discharge_power_kw + 1e-6, "discharge power <= max")


def test_f_efficiency_accounting() -> None:
    print("Test F - SOC transition and revenue account for efficiency")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2)
    prices = np.concatenate([np.full(144, 0.01), np.full(144, 0.50)])
    frame, _ = _run(cfg, prices, "F")
    dt = cfg.interval_hours
    c0 = frame["charge_power_kw"].iloc[0]
    d0 = frame["discharge_power_kw"].iloc[0]
    expected_end = cfg.initial_energy_kwh + cfg.charge_efficiency * dt * c0 - dt / cfg.discharge_efficiency * d0
    check(abs(frame["soc_kwh"].iloc[0] - expected_end) < 1e-6, "first SOC transition correct")
    # round-trip: energy out / energy in must be <= RTE for a full cycle
    grid_in = frame["charge_energy_kwh"].sum()
    grid_out = frame["discharge_energy_kwh"].sum()
    check(grid_in > 0 and grid_out > 0, "both charge and discharge occur")
    check(grid_out <= grid_in * cfg.ROUND_TRIP_EFFICIENCY + 1e-6, "RTE respected over the run")


def test_g_terminal_soc() -> None:
    print("Test G - equal_initial forces final SOC to initial SOC")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2)
    prices = np.concatenate([np.full(100, 0.01), np.full(188, 0.50)])
    frame, _ = _run(cfg, prices, "G")
    check(abs(frame["soc_kwh"].iloc[-1] - cfg.initial_energy_kwh) < 1e-6, "final SOC = initial")

    cfg_free = replace(CONFIG, TERMINAL_SOC_MODE="free", MODE=2)
    frame_free, _ = _run(cfg_free, prices, "G-free")
    check(frame_free["soc_kwh"].iloc[-1] < cfg_free.initial_energy_kwh - 1e-6, "free mode may end lower")


def test_h_mode1_daily_cycle_cap() -> None:
    print("Test H - Mode 1 daily EFC cap is respected")
    cfg = replace(CONFIG, MODE=1, CYCLES_PER_DAY=1.0, TERMINAL_SOC_MODE="equal_initial")
    # Two days of alternating cheap/expensive; the cap must bind per day.
    prices = np.tile(np.concatenate([np.full(72, -0.10), np.full(72, 0.80)]), 2)
    tidy = _series(prices)
    day_index = pd.factorize(tidy["day"])[0]
    result = solve_schedule(cfg, prices, day_index)
    frame = build_schedule_frame(cfg, tidy, result)
    validate_schedule(cfg, frame)
    batt_dis = frame["battery_discharge_kwh"]
    daily = frame.assign(bd=batt_dis).groupby("day")["bd"].sum() / cfg.usable_energy_kwh
    check((daily <= cfg.CYCLES_PER_DAY + 1e-6).all(), "daily EFC <= cycles_per_day")
    check(daily.min() > 0.5, "cap is actively used (revenue-seeking)")


def main() -> int:
    print("=" * 70)
    print("BESS TOS synthetic sanity tests")
    print("=" * 70)
    for test in (
        test_a_constant_price,
        test_b_single_price_spike,
        test_c_negative_price,
        test_d_soc_limits,
        test_e_power_limits,
        test_f_efficiency_accounting,
        test_g_terminal_soc,
        test_h_mode1_daily_cycle_cap,
    ):
        test()
        print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED")
        return 1
    print("RESULT: all sanity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
