"""Synthetic sanity tests for the TOS engine.

These run before any reliance on real NEM data and check that the optimiser and
its accounting behave correctly on problems with a known answer. Run with:

    python sanity_tests.py

Exits non-zero if any test fails.

Coverage
--------
A  constant price                 -> no artificial arbitrage
B  cheap then expensive           -> charge low, discharge high
C  negative price                 -> paid to charge
D  SOC limits                     -> never breached
E  power limits                   -> never breached
F  efficiency accounting          -> SOC transition + RTE
G  terminal SOC                   -> equal_initial / free
H  Mode 1 daily cycle cap         -> respected and actively used
I  exact optimum                  -> revenue equals a hand-computed value
J  MILP vs LP relaxation          -> MILP blocks the negative-price loophole and
                                     never exceeds the LP upper bound
K  power-rating side              -> battery-side current respects the C-rate /
                                     cable limit in both directions
L  Mode 1 cycle bases             -> each basis binds exactly at the cap
M  data pipeline                  -> coverage, interval-length and duplicate
                                     handling behave as documented
N  grid connection limits         -> phases/current capacity, static export and
                                     import caps, and validator enforcement
O  Amber subscription tiers       -> annualised usage tier boundaries and
                                     flat monthly fees per billed month
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, Config, REVENUE_TOL_USD
from helper import DataValidationError, prepare_price_series, read_aemo_mms_file
from optimizer import build_schedule_frame, solve_lp_bound, solve_schedule
from reporting import _calendar_months, amber_subscription_estimate
from validation import ConstraintViolationError, validate_schedule

FAILURES: list[str] = []


def _series(prices_kwh: np.ndarray, minutes: float = 5.0) -> pd.DataFrame:
    """Build a tidy price frame from an array of $/kWh prices."""
    n = len(prices_kwh)
    interval_start = pd.date_range("2026-01-01 00:00", periods=n, freq=f"{minutes}min")
    days = interval_start.normalize()
    return pd.DataFrame(
        {
            "timestamp": interval_start + pd.Timedelta(minutes=minutes),
            "interval_start": interval_start,
            "region": "TEST",
            "price_mwh": np.asarray(prices_kwh) * 1000.0,
            "price_kwh": np.asarray(prices_kwh, dtype=float),
            "day": days,
        }
    )


def _run(config: Config, prices_kwh: np.ndarray, label: str, minutes: float = 5.0):
    tidy = _series(prices_kwh, minutes)
    day_index = pd.factorize(tidy["day"])[0]
    result = solve_schedule(config, np.asarray(prices_kwh, float), day_index)
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
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2, COMPUTE_LP_BOUND=False)
    frame, _ = _run(cfg, np.full(288, 0.08), "A")
    check(abs(frame["interval_revenue"].sum()) < 1e-6, "net revenue is zero")
    check((frame["charge_power_kw"].abs() < 1e-6).all(), "no charging occurs")
    check((frame["discharge_power_kw"].abs() < 1e-6).all(), "no discharging occurs")


def test_b_single_price_spike() -> None:
    print("Test B - cheap then expensive: charge low, discharge high")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2, COMPUTE_LP_BOUND=False)
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
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2, COMPUTE_LP_BOUND=False)
    prices = np.concatenate([np.full(48, -0.05), np.full(240, 0.02)])
    frame, _ = _run(cfg, prices, "C")
    neg = frame["price"] < 0
    check(frame.loc[neg, "charge_power_kw"].sum() > 0, "charges during negative prices")
    check(frame["interval_revenue"].sum() > 0, "positive net revenue")
    check((frame.loc[neg, "charge_cost"] <= 0).all(), "negative cost (paid to charge)")


def test_d_soc_limits() -> None:
    print("Test D - SOC never breaches configured limits")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="free", MODE=2, COMPUTE_LP_BOUND=False)
    prices = np.concatenate([np.full(96, -0.20), np.full(96, 0.40)])
    frame, _ = _run(cfg, prices, "D")
    check(frame["soc_percent"].min() >= cfg.MIN_SOC_PERCENT - 1e-6, "SOC >= min")
    check(frame["soc_percent"].max() <= cfg.MAX_SOC_PERCENT + 1e-6, "SOC <= max")


def test_e_power_limits() -> None:
    print("Test E - charge/discharge power respects the meter-side limits")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="free", MODE=2, COMPUTE_LP_BOUND=False)
    prices = np.concatenate([np.full(96, -0.20), np.full(96, 0.40)])
    frame, _ = _run(cfg, prices, "E")
    check(frame["charge_power_kw"].max() <= cfg.effective_charge_power_kw + 1e-6, "charge power <= effective max")
    check(frame["discharge_power_kw"].max() <= cfg.effective_discharge_power_kw + 1e-6, "discharge power <= effective max")


def test_f_efficiency_accounting() -> None:
    print("Test F - SOC transition and revenue account for efficiency")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2, COMPUTE_LP_BOUND=False)
    prices = np.concatenate([np.full(144, 0.01), np.full(144, 0.50)])
    frame, _ = _run(cfg, prices, "F")
    dt = cfg.interval_hours
    c0 = frame["charge_power_kw"].iloc[0]
    d0 = frame["discharge_power_kw"].iloc[0]
    expected_end = cfg.initial_energy_kwh + cfg.charge_efficiency * dt * c0 - dt / cfg.discharge_efficiency * d0
    check(abs(frame["soc_kwh"].iloc[0] - expected_end) < 1e-6, "first SOC transition correct")
    grid_in = frame["charge_energy_kwh"].sum()
    grid_out = frame["discharge_energy_kwh"].sum()
    check(grid_in > 0 and grid_out > 0, "both charge and discharge occur")
    check(grid_out <= grid_in * cfg.ROUND_TRIP_EFFICIENCY + 1e-6, "RTE respected over the run")


def test_g_terminal_soc() -> None:
    print("Test G - equal_initial forces final SOC to initial SOC")
    cfg = replace(CONFIG, TERMINAL_SOC_MODE="equal_initial", MODE=2, COMPUTE_LP_BOUND=False)
    prices = np.concatenate([np.full(100, 0.01), np.full(188, 0.50)])
    frame, _ = _run(cfg, prices, "G")
    check(abs(frame["soc_kwh"].iloc[-1] - cfg.initial_energy_kwh) < 1e-6, "final SOC = initial")

    cfg_free = replace(cfg, TERMINAL_SOC_MODE="free")
    frame_free, _ = _run(cfg_free, prices, "G-free")
    check(frame_free["soc_kwh"].iloc[-1] < cfg_free.initial_energy_kwh - 1e-6, "free mode may end lower")


def test_h_mode1_daily_cycle_cap() -> None:
    print("Test H - Mode 1 daily EFC cap is respected")
    cfg = replace(CONFIG, MODE=1, CYCLES_PER_DAY=1.0, TERMINAL_SOC_MODE="equal_initial",
                  COMPUTE_LP_BOUND=False)
    prices = np.tile(np.concatenate([np.full(72, -0.10), np.full(72, 0.80)]), 2)
    tidy = _series(prices)
    day_index = pd.factorize(tidy["day"])[0]
    result = solve_schedule(cfg, prices, day_index)
    frame = build_schedule_frame(cfg, tidy, result)
    validate_schedule(cfg, frame)
    counted = cfg.mode1_cycle_coeff_c * frame["charge_power_kw"] + cfg.mode1_cycle_coeff_d * frame["discharge_power_kw"]
    daily = frame.assign(bd=counted).groupby("day")["bd"].sum() / cfg.usable_energy_kwh
    check((daily <= cfg.CYCLES_PER_DAY + 1e-6).all(), "daily EFC <= cycles_per_day")
    check(daily.min() > 0.5, "cap is actively used (revenue-seeking)")


def test_i_exact_optimum() -> None:
    print("Test I - revenue equals a hand-computed optimum (2 intervals)")
    cfg = replace(CONFIG, MODE=2, TERMINAL_SOC_MODE="equal_initial", COMPUTE_LP_BOUND=False)
    p_cheap, p_dear = 0.01, 0.50
    dt = cfg.interval_hours
    # Optimum: charge at the meter-side limit for one interval, then discharge the
    # stored battery-side energy back to the initial SOC.
    c = cfg.max_charge_power_kw
    battery_energy = cfg.charge_efficiency * dt * c
    d = battery_energy * cfg.discharge_efficiency / dt
    expected = p_dear * d * dt - p_cheap * c * dt
    frame = None
    tidy = _series(np.array([p_cheap, p_dear]))
    res = solve_schedule(cfg, np.array([p_cheap, p_dear]), pd.factorize(tidy["day"])[0])
    frame = build_schedule_frame(cfg, tidy, res)
    got = float(frame["interval_revenue"].sum())
    check(abs(got - expected) < REVENUE_TOL_USD * 1e3, f"revenue {got:.9f} == hand {expected:.9f}")
    check(abs(frame["charge_power_kw"].iloc[0] - c) < 1e-6, "charges at the meter-side limit")
    check(abs(frame["discharge_power_kw"].iloc[1] - d) < 1e-6, "discharges RTE x charge power")


def test_j_milp_vs_lp() -> None:
    print("Test J - MILP blocks the negative-price loophole and stays below the LP bound")
    prices = np.array([-0.06] * 48 + [0.02] * 240)
    milp_cfg = replace(CONFIG, MODE=2, TERMINAL_SOC_MODE="equal_initial", COMPUTE_LP_BOUND=False)
    lp_cfg = replace(milp_cfg, ENFORCE_NO_SIMULTANEOUS=False)
    frame_milp, _ = _run(milp_cfg, prices, "J-milp")
    frame_lp, _ = _run(lp_cfg, prices, "J-lp")
    c = frame_milp["charge_power_kw"].to_numpy()
    d = frame_milp["discharge_power_kw"].to_numpy()
    check(not ((c > 1e-9) & (d > 1e-9)).any(), "MILP has no simultaneous flows")
    sim_lp = int(((frame_lp["charge_power_kw"] > 1e-9) & (frame_lp["discharge_power_kw"] > 1e-9)).sum())
    check(sim_lp > 0, f"LP relaxation does exploit simultaneity ({sim_lp} intervals)")
    tidy = _series(prices)
    day_index = pd.factorize(tidy["day"])[0]
    bound = solve_lp_bound(milp_cfg, prices, day_index)
    check(bound + 1e-9 >= float(frame_milp["interval_revenue"].sum()), "MILP revenue <= LP bound")
    check(float(frame_lp["interval_revenue"].sum()) > float(frame_milp["interval_revenue"].sum()),
          "LP revenue is strictly optimistic on this instance")


def test_k_power_rating_side() -> None:
    print("Test K - battery-side current respects the DC rating in both directions")
    cfg = replace(CONFIG, MODE=2, TERMINAL_SOC_MODE="free", COMPUTE_LP_BOUND=False)
    check(abs(cfg.max_charge_power_kw - cfg.dc_charge_power_kw / cfg.charge_efficiency) < 1e-12,
          "meter-side charge limit = DC rating / eta_c")
    check(abs(cfg.max_discharge_power_kw - cfg.dc_discharge_power_kw * cfg.discharge_efficiency) < 1e-12,
          "meter-side discharge limit = DC rating x eta_d")
    prices = np.concatenate([np.full(96, -0.30), np.full(96, 0.90)])
    frame, _ = _run(cfg, prices, "K")
    battery_charge = cfg.charge_efficiency * frame["charge_power_kw"]
    battery_discharge = frame["discharge_power_kw"] / cfg.discharge_efficiency
    check(battery_charge.max() <= cfg.dc_charge_power_kw + 1e-6,
          f"battery-side charge {battery_charge.max():.4f} <= rating {cfg.dc_charge_power_kw:.4f} kW")
    check(battery_discharge.max() <= cfg.dc_discharge_power_kw + 1e-6,
          f"battery-side discharge {battery_discharge.max():.4f} <= rating {cfg.dc_discharge_power_kw:.4f} kW")
    amps = battery_discharge.max() / cfg.NOMINAL_VOLTAGE_V * 1000.0
    check(amps <= min(cfg.MAX_DISCHARGE_C_RATE * cfg.NOMINAL_CAPACITY_AH, cfg.MAX_CURRENT_A) + 1e-6,
          f"battery-side discharge current {amps:.2f} A <= effective limit")


def test_l_mode1_cycle_bases() -> None:
    print("Test L - Mode 1 cap binds exactly for every cycle basis")
    prices = np.linspace(0.01, 0.90, 24)
    for basis in ("battery_discharge", "battery_charge", "grid_discharge", "grid_charge"):
        cfg = replace(CONFIG, MODE=1, CYCLES_PER_DAY=0.2, TERMINAL_SOC_MODE="free",
                      MODE1_CYCLE_BASIS=basis, INTERVAL_MINUTES=60.0, COMPUTE_LP_BOUND=False)
        frame, _ = _run(cfg, prices, f"L-{basis}", minutes=60.0)
        counted = float((cfg.mode1_cycle_coeff_c * frame["charge_power_kw"]
                         + cfg.mode1_cycle_coeff_d * frame["discharge_power_kw"]).sum())
        cap = cfg.CYCLES_PER_DAY * cfg.usable_energy_kwh
        check(abs(counted - cap) < 1e-6, f"{basis}: counted {counted:.6f} == cap {cap:.6f}")


def test_n_grid_connection_limits() -> None:
    print("Test N - AC connection capacity and static export/import caps are enforced")
    # 1) Capacity scales with phases at the same per-phase current.
    one = replace(CONFIG, PHASES=1, MAX_CURRENT_A_PER_PHASE=32.0,
                  MAX_EXPORT_KW=None, MAX_IMPORT_KW=None)
    three = replace(CONFIG, PHASES=3, MAX_CURRENT_A_PER_PHASE=32.0,
                    MAX_EXPORT_KW=None, MAX_IMPORT_KW=None)
    check(abs(three.connection_power_kw - 3.0 * one.connection_power_kw) < 1e-9,
          "3-phase capacity = 3 x 1-phase capacity")
    check(abs(one.connection_power_kw - 7.36) < 1e-9, "1-phase 32 A/phase = 7.36 kW")

    prices = np.concatenate([np.full(96, -0.20), np.full(96, 0.40)])

    # 2) A static export cap is binding and respected.
    cfg = replace(CONFIG, MODE=2, TERMINAL_SOC_MODE="free", COMPUTE_LP_BOUND=False,
                  MAX_EXPORT_KW=4.5, MAX_IMPORT_KW=None)
    frame, _ = _run(cfg, prices, "N-export")
    check(abs(cfg.grid_export_limit_kw - 4.5) < 1e-9, "static export cap = 4.5 kW")
    check(frame["discharge_power_kw"].max() <= cfg.effective_discharge_power_kw + 1e-6,
          f"discharge {frame['discharge_power_kw'].max():.4f} kW <= "
          f"effective {cfg.effective_discharge_power_kw:.4f} kW")

    # 3) A 1-phase connection can be the binding charge (import) limit.
    cfg_in = replace(CONFIG, MODE=2, TERMINAL_SOC_MODE="free", COMPUTE_LP_BOUND=False,
                     PHASES=1, MAX_CURRENT_A_PER_PHASE=10.0)
    frame_in, _ = _run(cfg_in, prices, "N-import")
    check(abs(cfg_in.effective_charge_power_kw - 2.3) < 1e-9,
          "1-phase 10 A/phase import limit = 2.3 kW")
    check(frame_in["charge_power_kw"].max() <= cfg_in.effective_charge_power_kw + 1e-6,
          f"charge {frame_in['charge_power_kw'].max():.4f} kW <= "
          f"effective {cfg_in.effective_charge_power_kw:.4f} kW")

    # 4) The validator rejects a schedule that breaches the grid export cap.
    frame_bad = frame.copy()
    frame_bad.loc[0, "discharge_power_kw"] = cfg.effective_discharge_power_kw + 1.0
    try:
        validate_schedule(cfg, frame_bad)
        check(False, "export-cap breach must fail validation")
    except ConstraintViolationError as exc:
        check("discharge_power_within_grid_export_limit" in str(exc),
              "validator flags the export-cap breach")


def test_o_amber_subscription() -> None:
    print("Test O - Amber subscription tiers and flat monthly-fee billing")
    year = 365.25
    cases = (
        (10_000.0, 25.0), (10_001.0, 50.0),
        (20_000.0, 50.0), (20_001.0, 125.0),
        (50_000.0, 125.0), (50_001.0, 200.0),
        (100_000.0, 200.0), (100_001.0, 350.0),
    )
    for usage, fee in cases:
        est = amber_subscription_estimate(usage, year)
        check(abs(est["monthly_fee"] - fee) < 1e-9,
              f"{usage:,.0f} kWh/yr -> ${fee:,.0f}/month")

    # A one-month period pays exactly one flat monthly fee (no day proration).
    est = amber_subscription_estimate(3_500.0, 31.0, 1)
    expected_annual = 3_500.0 * 365.25 / 31.0
    check(abs(est["annual_usage_kwh"] - expected_annual) < 1e-9,
          f"31 days of 3,500 kWh -> {expected_annual:,.0f} kWh/yr")
    check(abs(est["monthly_fee"] - 125.0) < 1e-9, "lands in the $125/month tier")
    check(abs(est["period_cost"] - 125.0) < 1e-9,
          "31 days in one month costs 1 x $125 (not prorated)")

    two_months = amber_subscription_estimate(7_000.0, 61.0, 2)
    check(abs(two_months["period_cost"] - 2.0 * two_months["monthly_fee"]) < 1e-9,
          "two billed months cost 2 x the monthly fee")

    full = amber_subscription_estimate(100_000.0, year, 12)
    check(abs(full["period_cost"] - 200.0 * 12.0) < 1e-9,
          "a full year at the $200 tier costs 12 x the monthly fee")

    # Calendar-month counting: an inclusive range touches every month it spans.
    check(_calendar_months("2026-08-01", "2026-08-31") == 1, "Aug 1-31 -> 1 month")
    check(_calendar_months("2026-08-15", "2026-09-14") == 2, "Aug 15-Sep 14 -> 2 months")
    check(_calendar_months("2026-01-01", "2026-12-31") == 12, "full year -> 12 months")

    try:
        amber_subscription_estimate(1_000.0, 0.0)
        check(False, "zero-length period must raise")
    except ValueError:
        check(True, "zero-length period rejected")
    try:
        amber_subscription_estimate(1_000.0, 31.0, 0)
        check(False, "zero billing months must raise")
    except ValueError:
        check(True, "zero billing months rejected")


def _write_mms(path: Path, start: pd.Timestamp, intervals: int, interval_min: int,
               region: str = "VIC1", price: float = 50.0,
               extra_rows: list[tuple[str, float]] | None = None,
               rows_override: list[list[str]] | None = None,
               field_count_bad: bool = False) -> None:
    """Write a minimal but structurally valid AEMO MMS DISPATCHPRICE file."""
    header = (f'I,DISPATCH,PRICE,{interval_min},SETTLEMENTDATE,RUNNO,REGIONID,INTERVENTION,RRP')
    lines = ["C,SETP.WORLD,DVD_DISPATCHPRICE,AEMO,PUBLIC,2026/09/08,13:56:34,1,MONTHLY_ARCHIVE,1", header]
    n = intervals if rows_override is None else len(rows_override)
    for i in range(n):
        if rows_override is not None:
            fields = rows_override[i]
        else:
            end = start + pd.Timedelta(minutes=interval_min * (i + 1))
            fields = ["D", "DISPATCH", "PRICE", str(interval_min),
                      f'"{end:%Y/%m/%d %H:%M:%S}"', "1", region, "0", f"{price + i:.5f}"]
        if field_count_bad and i == 1:
            fields = fields[:-1]
        lines.append(",".join(fields))
    lines.append('C,"END OF REPORT",%d' % (len(lines) + 1))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_m_data_pipeline() -> None:
    print("Test M - data pipeline: coverage, interval length, duplicate resolution")
    tmp = Path(tempfile.mkdtemp(prefix="bess_tos_test_"))

    # M1: header interval disagrees with INTERVAL_MINUTES -> hard error
    f1 = tmp / "m1.csv"
    _write_mms(f1, pd.Timestamp("2026-01-01 00:00"), 288, 30)
    cfg = replace(CONFIG, DATA_DIR=tmp, START_DATE="2026-01-01", END_DATE="2026-01-01",
                  COMPUTE_LP_BOUND=False)
    try:
        prepare_price_series(cfg)
        check(False, "30-min file must not be priced as 5-min")
    except DataValidationError as exc:
        check("interval" in str(exc).lower(), f"30-min file rejected: {str(exc)[:60]}...")

    # M2: truncated series -> hard error by default, warning when opted out
    f2 = tmp / "m2.csv"
    f1.unlink()
    _write_mms(f2, pd.Timestamp("2026-01-01 00:00"), 144, 5)   # half a day
    cfg = replace(cfg, END_DATE="2026-01-02")
    try:
        prepare_price_series(cfg)
        check(False, "short coverage must raise with REQUIRE_FULL_COVERAGE=True")
    except DataValidationError as exc:
        check("missing" in str(exc).lower(), "short coverage raises by default")
    cfg_warn = replace(cfg, REQUIRE_FULL_COVERAGE=False)
    tidy, rep = prepare_price_series(cfg_warn)
    check(rep.n_missing_intervals > 0 and bool(rep.warnings), "opt-out reports the gap as a warning")
    check(rep.n_missing_tail > 0, "tail truncation is identified as such")

    # M3: duplicate (timestamp, region) -> highest INTERVENTION wins
    f2.unlink()
    f3 = tmp / "m3.csv"
    rows = [
        ["D", "DISPATCH", "PRICE", "5", '"2026/01/01 00:05:00"', "1", "VIC1", "0", "11.00000"],
        ["D", "DISPATCH", "PRICE", "5", '"2026/01/01 00:05:00"', "1", "VIC1", "1", "99.00000"],
    ]
    _write_mms(f3, pd.Timestamp("2026-01-01 00:00"), 1, 5, rows_override=rows)
    cfg_one = replace(CONFIG, DATA_DIR=tmp, START_DATE="2026-01-01", END_DATE="2026-01-01",
                      REQUIRE_FULL_COVERAGE=False, COMPUTE_LP_BOUND=False)
    tidy, rep = prepare_price_series(cfg_one)
    check(rep.n_duplicate_timestamps == 1, "duplicate counted")
    check(abs(tidy["price_mwh"].iloc[0] - 99.0) < 1e-9,
          "intervention run (highest INTERVENTION) wins the duplicate")
    check(tidy["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:05:00"),
          "duplicate timestamp is parsed as interval-ending")

    # M4: malformed field count -> hard error, not silent column shift
    f3.unlink()
    f4 = tmp / "m4.csv"
    _write_mms(f4, pd.Timestamp("2026-01-01 00:00"), 3, 5, field_count_bad=True)
    try:
        read_aemo_mms_file(f4)
        check(False, "malformed row must raise")
    except DataValidationError as exc:
        check("fields" in str(exc).lower(), f"malformed row rejected: {str(exc)[:60]}...")

    for path in tmp.glob("*"):
        path.unlink()
    tmp.rmdir()


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
        test_i_exact_optimum,
        test_j_milp_vs_lp,
        test_k_power_rating_side,
        test_l_mode1_cycle_bases,
        test_m_data_pipeline,
        test_n_grid_connection_limits,
        test_o_amber_subscription,
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
