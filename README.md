# bess_TOS — Theoretically Optimal Strategy for BESS energy arbitrage in the NEM

A perfect-hindsight (**oracle**) benchmark that computes the maximum possible
gross energy-arbitrage revenue for a Battery Energy Storage System (BESS)
trading in the Australian National Electricity Market (NEM).

> **This is a benchmark, not a trading strategy.** The optimiser is given the
> complete historical 5-minute settlement-price series before scheduling, so it
> knows the future. It must not be described as deployable trading logic. Its
> purpose is to establish an upper bound against which forecasting / real-time
> strategies can be compared.

## Quick start

```bash
pip install -r requirements.txt
python main.py          # runs Mode 2 and Mode 1 for the configured period
python sanity_tests.py  # synthetic correctness tests (A–H)
```

Outputs land in `outputs/` (interval-level CSV + summary text), plus
`generated_plot.png` (four-panel diagnostic) and `generated_energy_plot.png`
(cumulative grid energy in kWh, for retailer access-tier selection).

## Configuration

Everything user-facing lives in `config.py` (the `Config` dataclass / `CONFIG`
object). Change region, dates, battery parameters, efficiency, initial/terminal
SOC and the optimisation mode there — never in the implementation.

| Setting | Meaning |
|---|---|
| `REGION` | `NSW1`, `QLD1`, `SA1`, `TAS1`, `VIC1` |
| `START_DATE`, `END_DATE` | inclusive calendar dates (NEM time) |
| `BATTERY_CAPACITY_KWH` | nominal energy capacity (kWh) |
| `NOMINAL_VOLTAGE_V` | nominal pack voltage (V) |
| `NOMINAL_CAPACITY_AH` | nominal pack capacity (Ah) — must satisfy `Ah·V/1000 ≈ kWh` |
| `MIN_SOC_PERCENT`, `MAX_SOC_PERCENT` | usable SOC window |
| `INITIAL_SOC_PERCENT` | starting SOC |
| `MAX_CHARGE_C_RATE`, `MAX_DISCHARGE_C_RATE` | C-rates (definition below) |
| `MAX_CURRENT_A` | hard cable-size current limit (A), applies to charge and discharge |
| `ROUND_TRIP_EFFICIENCY` | split symmetrically (see below) |
| `CHARGE_EFFICIENCY`, `DISCHARGE_EFFICIENCY` | optional independent overrides |
| `MODE` | `1` = fixed daily cycle cap, `2` = free timing |
| `CYCLES_PER_DAY` | EFC cap per calendar day (Mode 1) |
| `TERMINAL_SOC_MODE` | `equal_initial` or `free` |
| `ENFORCE_NO_SIMULTANEOUS` | `True` = MILP (correct); `False` = LP upper bound |

### Power limits from C-rate, current and voltage

A C-rate is defined against the **nominal Ah capacity**, giving a current; the
current becomes power via the nominal voltage:

```
I_c_rate   = C-rate × NOMINAL_CAPACITY_AH
I_effective = min(I_c_rate, MAX_CURRENT_A)        # cable limit tightens only
P_max       = I_effective × NOMINAL_VOLTAGE_V / 1000   # kW
```

`config.validate()` rejects a battery specification where
`NOMINAL_CAPACITY_AH × NOMINAL_VOLTAGE_V / 1000` disagrees with
`BATTERY_CAPACITY_KWH` beyond tolerance, so the three values stay coherent.
The summary reports the effective charge/discharge current and whether each is
C-rate- or cable-limited.

`main.py` runs **both** modes for the configured period and writes one CSV and
one summary per mode.

## Optimisation model

Per 5-minute interval `t`: grid-side charge power `c[t]`, grid-side discharge
power `d[t]`, a mode binary `y[t]`, and battery-side energy `E[t]`.

```
maximise   Σ  Δt · price[t] · (d[t] − c[t])                 (Δt = 1/12 h)
subject to E[t+1] = E[t] + η_c·Δt·c[t] − (Δt/η_d)·d[t]
           0 ≤ c[t] ≤ P_charge_max ,   0 ≤ d[t] ≤ P_discharge_max
           E_min ≤ E[k] ≤ E_max ,      E[0] = initial energy
           E[T] = E[0]  (equal_initial) or free within limits
           c[t] ≤ P_charge_max·y[t] ,  d[t] ≤ P_discharge_max·(1 − y[t])
           Σ_{t in day D} (Δt/η_d)·d[t] ≤ CYCLES_PER_DAY · usable_energy   (Mode 1)
```

`P_charge_max`/`P_discharge_max` are the effective powers from
`min(C-rate × Ah, MAX_CURRENT_A) × voltage / 1000` (see above).

* **Efficiency.** `η_c = η_d = √ROUND_TRIP_EFFICIENCY` (symmetric split). Only
  one conversion is applied per direction; the round trip is `η_c·η_d = RTE`.
* **Energy conventions.** `c`/`d` are grid/meter-side (what the meter settles);
  `E` (and `soc_kwh`) is battery-side.
* **Equivalent Full Cycles (EFC).** `EFC = Σ (battery-side discharge kWh) /
  usable_energy`. Mode 1 caps daily battery-side discharge throughput at
  `CYCLES_PER_DAY × usable_energy`.
* **Terminal SOC.** `equal_initial` prevents the benchmark from manufacturing
  revenue by finishing with an empty battery; `free` is available for
  comparison.
* **Negative prices** are preserved. They create a genuine paid-to-charge
  incentive.

### Why a MILP (and not just an LP)

Without binaries the LP can exploit a **non-physical** loophole at negative
prices: it charges and discharges simultaneously, holding SOC constant while
being paid for the energy routed through round-trip losses. Reducing both flows
keeping SOC fixed changes revenue by `Δt·price·δ·(1 − RTE)`, which is positive
when `price < 0`. A real battery has a single AC port, so this is impossible.

The binaries (`c ≤ P·y`, `d ≤ P·(1−y)`) remove it. HiGHS (`scipy.optimize.milp`)
proves global optimality. Setting `ENFORCE_NO_SIMULTANEOUS = False` reverts to
the LP, which is a valid but optimistic **upper bound** (its objective exceeds
the physical optimum).

## Data pipeline (`helper.py`)

Reusable and free of BESS logic:

`discover_price_files` → `read_aemo_mms_file` → `combine_data` → `clean_data` →
`filter_region` → `filter_date_range` → `check_completeness` →
`prepare_price_series`.

It parses AEMO MMS `DISPATCHPRICE` archives (the four leading non-column fields
are handled automatically), supports multiple monthly files, removes duplicate
`(timestamp, region)` records (e.g. later intervention runs) and reports gaps
without imputing them.

### Time conventions

* Timestamps are **interval-ending** and in **NEM time (AEST, UTC+10, no
  daylight saving)**; naive datetimes are therefore correct.
* Daily grouping uses `interval_start = timestamp − 5 min`, so the interval
  ending at 00:00 correctly belongs to the previous calendar day.
* Date ranges are inclusive of the whole `END_DATE`.
* Each expected 5-minute interval is validated against the timestamps present;
  row counts are never assumed.

## Output

Interval-level DataFrame / CSV columns: `timestamp, region, price_mwh, price,
charge_power_kw, discharge_power_kw, charge_energy_kwh, discharge_energy_kwh,
battery_charge_kwh, battery_discharge_kwh, soc_start_kwh, soc_kwh, soc_percent,
charge_cost, discharge_revenue, interval_revenue, cumulative_revenue,
cumulative_charge_energy_kwh, cumulative_discharge_energy_kwh, operating_state,
equivalent_full_cycles, day`.

The printed summary covers configuration, data quality, performance totals,
charge/discharge event statistics, monthly grid energy usage (total import,
export and net, for retailer access-tier selection), calendar-day and
rolling-24-hour performance, and optimiser diagnostics (solver, status,
objective, runtime, global-optimality, constraint violations).

## Validation and tests

* `validation.py` independently recomputes the SOC trajectory, energy flows,
  revenue and EFC from the schedule and raises on any violation — it never
  trusts the solver.
* `sanity_tests.py`: A constant price, B price spike, C negative price, D SOC
  limits, E power limits, F efficiency accounting, G terminal SOC, H Mode 1
  daily cycle cap.

## Project layout

```
bess_TOS/
├── main.py            orchestration (pipeline steps 1–12)
├── config.py          all user settings + derived quantities
├── helper.py          reusable NEM data pipeline
├── optimizer.py       (MI)LP model, HiGHS solve, schedule frame
├── validation.py      independent constraint/revenue checks
├── reporting.py       performance metrics + summary formatting
├── visualization.py   four-panel + cumulative-revenue plots
├── sanity_tests.py    synthetic tests A–H
├── requirements.txt
├── generated_plot.png       4-panel diagnostic
├── generated_energy_plot.png cumulative grid energy (kWh)
├── data/              AEMO DISPATCHPRICE archives
└── outputs/           per-mode CSV + summary text + plots
```

## Scope / deliberately excluded constraints

Not requested and therefore **not** implemented, to keep a clean baseline:
minimum charge/discharge duration, ramp-rate limits, grid-connection/export
limits, auxiliary/standby consumption, degradation or cycle-cost penalties,
reserve requirements, and price forecasting. Each can be added as an
additional linear constraint without changing the architecture.
