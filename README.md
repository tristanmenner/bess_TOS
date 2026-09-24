# bess_TOS — Theoretically Optimal Strategy (TOS) for BESS energy arbitrage in the NEM

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
python sanity_tests.py  # synthetic correctness tests A–M
```

Outputs land in `outputs/` (interval-level CSV + summary text + data manifest),
plus `generated_plot.png` (four-panel diagnostic), `generated_energy_plot.png`
(cumulative grid energy in kWh, for retailer access-tier selection) and one
`cumulative_revenue_modeX.png` per mode with the Amber break-even line.

Every setting can also be overridden from the command line, so every committed
artefact is reproducible:

```bash
# the 119 kWh pack (outputs/119kWh and outputs/119kWh_v2 are pre-connection-cap archives)
python main.py --capacity-kwh 119 --voltage 743.6 --ah 160 --outdir outputs/119kWh

# exact-optimality run (proves the MIP gap is zero; slower on large packs)
python main.py --mip-gap 0

# optimistic LP upper bound (non-physical; clearly labelled in the summary)
python main.py --lp-relaxation --mode 2
```

`python main.py --help` lists all overrides (region, dates, pack spec, SOC
window, efficiencies, C-rates, current limit, phases and connection current,
static import/export caps, cycle basis, day basis, MIP gap, coverage policy,
output directory).

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
| `MIN_SOC_PERCENT`, `MAX_SOC_PERCENT` | usable SOC window (`MIN` may be 0) |
| `INITIAL_SOC_PERCENT` | starting SOC |
| `MAX_CHARGE_C_RATE`, `MAX_DISCHARGE_C_RATE` | C-rates (definition below) |
| `MAX_CURRENT_A` | hard cable-size current limit (A), applies to charge and discharge |
| `PHASES` | AC connection phases: `1` or `3` |
| `NOMINAL_AC_VOLTAGE_PER_PHASE_V` | phase voltage used for the connection cap (230 V) |
| `MAX_CURRENT_A_PER_PHASE` | AC connection current limit per phase (A) |
| `MAX_EXPORT_KW`, `MAX_IMPORT_KW` | optional static site caps (kW); `None` = connection capacity |
| `ROUND_TRIP_EFFICIENCY` | split symmetrically (see below) |
| `CHARGE_EFFICIENCY`, `DISCHARGE_EFFICIENCY` | optional independent overrides |
| `MODE` | `1` = fixed daily cycle cap, `2` = free timing |
| `CYCLES_PER_DAY` | EFC cap per day (Mode 1) |
| `MODE1_CYCLE_BASIS` | which flow Mode 1 counts: `battery_discharge` (default), `battery_charge`, `grid_discharge`, `grid_charge` |
| `DAY_BASIS` | `calendar` (default) or `nem_trading` (04:00→04:00 NEM time) |
| `TERMINAL_SOC_MODE` | `equal_initial` or `free` |
| `ENFORCE_NO_SIMULTANEOUS` | `True` = MILP (correct); `False` = LP upper bound |
| `MIP_REL_GAP` | MILP gap target (default `1e-4`; `0` proves exact optimality) |
| `MIP_TIME_LIMIT_S` | optional MILP time limit (incumbent is kept if it is hit) |
| `COMPUTE_LP_BOUND` | also solve the LP relaxation as an optimality certificate |
| `REQUIRE_FULL_COVERAGE` | fail (default) instead of warning when the price series does not cover the period |

### Power limits from C-rate, current and voltage

A C-rate is defined against the **nominal Ah capacity**, giving a current; the
current becomes **battery-side (DC) power** via the nominal voltage:

```
I_c_rate    = C-rate × NOMINAL_CAPACITY_AH
I_effective = min(I_c_rate, MAX_CURRENT_A)          # cable limit tightens only
P_dc        = I_effective × NOMINAL_VOLTAGE_V / 1000   # kW, battery side
```

The optimiser constrains **meter-side** flows, so the DC rating is converted
through the converter efficiency — the battery-side current is what the C-rate
and cable actually limit:

```
c ≤ P_dc_charge / η_c          (meter-side charging)
d ≤ P_dc_discharge × η_d       (meter-side discharging)
```

Both the DC ratings and the meter-side limits are printed in the summary, and
`validation.py` re-checks that the battery-side power never exceeds the DC
rating. Applying the DC rating directly to the meter-side flow (as an earlier
revision did) allowed a discharge current 5.4 % above the stated limit.

### Grid connection limit

Separately from the battery/DC rating, the AC connection is limited to

```
P_connection = PHASES × MAX_CURRENT_A_PER_PHASE × NOMINAL_AC_VOLTAGE_PER_PHASE_V / 1000
```

and the optimiser applies the tighter of the battery/converter and grid limits:

```
P_charge_eff    = min(P_charge_max,  grid_import_limit_kw)
P_discharge_eff = min(P_discharge_max, grid_export_limit_kw)
```

`MAX_IMPORT_KW` / `MAX_EXPORT_KW` can tighten either side further — for example
the 4.5 kW static export default offered for a 30 kW solar connection. With no
site load or generation modelled, these bounds are exactly the net meter-flow
caps (charge = import, discharge = export). The summary prints the connection
capacity, both static caps and the effective limits, with which source binds.

Defaults model a 3-phase, 100 A/phase connection (69.0 kW). Scenario overrides:

```bash
# 100 kW-class negotiated connection (~145 A/phase)
python main.py --phases 3 --current-per-phase 145 --outdir outputs/100kW

# 30 kW basic connection (~43 A/phase) with a 4.5 kW static export cap
python main.py --phases 3 --current-per-phase 43 --max-export-kw 4.5 --outdir outputs/30kW_static

# single-phase 100 A/phase (23 kW)
python main.py --phases 1 --current-per-phase 100 --outdir outputs/1ph
```

`config.validate()` rejects a battery specification where
`NOMINAL_CAPACITY_AH × NOMINAL_VOLTAGE_V / 1000` disagrees with
`BATTERY_CAPACITY_KWH` beyond tolerance, so the three values stay coherent.

`main.py` runs **both** modes for the configured period and writes one CSV and
one summary per mode.

## Optimisation model

Per 5-minute interval `t`: meter-side charge power `c[t]`, meter-side discharge
power `d[t]`, a mode binary `y[t]`, and battery-side energy `E[t]`.

```
maximise   Σ  Δt · price[t] · (d[t] − c[t])                 (Δt = 1/12 h)
subject to E[t+1] = E[t] + η_c·Δt·c[t] − (Δt/η_d)·d[t]
           0 ≤ c[t] ≤ P_charge_eff ,   0 ≤ d[t] ≤ P_discharge_eff
           E_min ≤ E[k] ≤ E_max ,      E[0] = initial energy
           E[T] = E[0]  (equal_initial) or free within limits
           c[t] ≤ P_charge_eff·y[t] ,  d[t] ≤ P_discharge_eff·(1 − y[t])
           Σ_{t in day D} (k_c·c[t] + k_d·d[t]) ≤ CYCLES_PER_DAY · usable_energy   (Mode 1)
```

`P_charge_max = P_dc_charge/η_c` and `P_discharge_max = P_dc_discharge·η_d` are
the meter-side equivalents of the battery-side rating, tightened by the AC
connection: `P_charge_eff = min(P_charge_max, grid_import_limit_kw)` and
`P_discharge_eff = min(P_discharge_max, grid_export_limit_kw)` (see above).
The Mode 1 coefficients `k_c`, `k_d` implement `MODE1_CYCLE_BASIS`; the default
counts battery-side discharge, i.e. `k_d = Δt/η_d`, `k_c = 0`:

* **Efficiency.** `η_c = η_d = √ROUND_TRIP_EFFICIENCY` (symmetric split). Only
  one conversion is applied per direction; the round trip is `η_c·η_d = RTE`.
* **Energy conventions.** `c`/`d` are meter-side (what the meter settles);
  `E` (and `soc_kwh`) is battery-side.
* **Equivalent Full Cycles (EFC).** With the default basis,
  `EFC = Σ (battery-side discharge kWh) / usable_energy`; the Mode 1 daily cap
  therefore limits battery-side discharge throughput. Charging is not separately
  capped, but over a horizon with `E[T] = E[0]` total battery-side charge equals
  total battery-side discharge, so `CYCLES_PER_DAY × days` bounds the monthly
  throughput too. Use `MODE1_CYCLE_BASIS` if the specification being benchmarked
  defines the cycle differently.
* **Terminal SOC.** `equal_initial` prevents the benchmark from manufacturing
  revenue by finishing with an empty battery; `free` is available for
  comparison (and is worth roughly the initial stored energy times the price
  spread).
* **Negative prices** are preserved. They create a genuine paid-to-charge
  incentive (and, with `equal_initial`, a genuine incentive to cycle: you are
  paid more for importing than you pay for the smaller export).

### Why a MILP (and not just an LP)

Without binaries the LP can exploit a **non-physical** loophole at negative
prices: it charges and discharges simultaneously, holding SOC constant while
being paid for the energy routed through round-trip losses. Reducing both flows
keeping SOC fixed changes revenue by `Δt·price·δ·(1 − RTE)`, which is positive
when `price < 0`. A real battery has a single AC port, so this is impossible.

The binaries (`c ≤ P·y`, `d ≤ P·(1−y)`) remove it. HiGHS (`scipy.optimize.milp`)
proves optimality to the configured gap; the achieved gap and dual bound are
reported in every summary.

Setting `ENFORCE_NO_SIMULTANEOUS = False` (or `--lp-relaxation`) solves the LP
relaxation instead. Its **objective** is a valid optimistic upper bound on the
physical optimum; the interval schedule is *not* a dispatch plan and the summary
says so explicitly. The relaxation solution is reported as solved (no
canonicalisation), so the CSV revenue and the LP objective always agree.

### Optimality certificate

With `COMPUTE_LP_BOUND = True` (default) each MILP run also solves the LP
relaxation and prints:

```
MILP objective (this run)     : $72.754322
LP relaxation upper bound     : $72.756549
MILP gap vs LP bound          : $0.002226  (0.003060%)
bound check                   : PASSED (MILP <= LP bound)
```

This is an *independent* check that the MILP result cannot exceed the physical
optimum (a bug that over-constrained the model would show up as a failed bound
check, and the run would abort).

### Solver termination

`MIP_REL_GAP` defaults to `1e-4` (HiGHS' default): optimality is claimed only
when the *achieved* gap meets the target, and the achieved gap, dual bound and
node count are always printed. For small packs `--mip-gap 0` proves exact
optimality in seconds; for large packs (e.g. 119 kWh / 74 kW) exact proof can be
intractable, so raise the target or set `--mip-gap 1e-6`. If a time limit is hit
the incumbent schedule is kept and the summary states that optimality was not
proven.

## Data pipeline (`helper.py`)

Reusable and free of BESS logic:

`discover_price_files` → `read_aemo_mms_file` → `combine_data` → `clean_data` →
`filter_region` → `filter_date_range` → `check_completeness` →
`prepare_price_series`.

It parses AEMO MMS `DISPATCHPRICE` archives and:

* **validates the field count of every `D` row against the `I` header**, so a
  format change raises instead of silently shifting columns;
* **reads the MMS header's interval length** (5 for 5-minute settlement) and
  refuses to price a file whose interval disagrees with `INTERVAL_MINUTES`
  (e.g. a 30-minute-settlement file);
* resolves duplicate `(timestamp, region)` records by preferring the highest
  `INTERVENTION` flag, then the highest `RUNNO`, then the last record — i.e.
  the latest run wins;
* **validates coverage against the requested period**, not just the rows that
  happen to be present. `REQUIRE_FULL_COVERAGE=True` (default) raises if any
  expected interval is missing; otherwise every gap is reported (overall and as
  head/tail counts) in the summary;
* reports the SHA-256 of every input file and writes
  `outputs/data_manifest.sha256` for provenance;
* never imputes prices.

### Time conventions

* Timestamps are **interval-ending** and in **NEM time (AEST, UTC+10, no
  daylight saving)**; naive datetimes are therefore correct.
* `day` (used by the Mode 1 cap and all daily reporting) follows `DAY_BASIS`:
  `calendar` = midnight-to-midnight (default), `nem_trading` = the AEMO trading
  day, i.e. interval-starts from 04:00 to 03:55 the next day. With
  `nem_trading`, the first and last day of a period are partial days (reported).
* Date ranges are inclusive of the whole `END_DATE`.
* Each expected interval is validated against the requested window before the
  optimiser runs.

### Data provenance

The NEM archive used for the shipped outputs is a local file under `data/`
(deliberately git-ignored — 17 MB). `data/README.md` records its identity and
SHA-256, and every run re-prints the hash of the files it actually consumed into
the summary and `outputs/data_manifest.sha256`. **A clone without `data/` cannot
reproduce the numbers**; the manifest makes any substitution detectable.

## Output

Interval-level DataFrame / CSV columns: `timestamp, region, price_mwh, price,
charge_power_kw, discharge_power_kw, charge_energy_kwh, discharge_energy_kwh,
battery_charge_kwh, battery_discharge_kwh, soc_start_kwh, soc_kwh, soc_percent,
charge_cost, discharge_revenue, interval_revenue, cumulative_revenue,
cumulative_charge_energy_kwh, cumulative_discharge_energy_kwh, operating_state,
equivalent_full_cycles, day`.

`operating_state` is `charge`, `discharge` or `idle` for physical schedules; the
LP relaxation can additionally report `simultaneous` (non-physical).

The printed summary covers configuration, **data coverage and file hashes**,
performance totals, charge/discharge event statistics, monthly grid energy usage
(total import, export and net, for retailer access-tier selection), an **Amber
Electric subscription break-even reference**, daily and rolling-24-hour
performance, optimiser diagnostics (solver, status, objective, runtime,
achieved MIP gap, dual bound, node count, power-bound violation, constraint
violations), and the **optimality certificate** (LP relaxation upper bound vs
MILP objective).

### Break-even overlay (Amber Electric)

Both cumulative-revenue charts (the standalone `cumulative_revenue_modeX.png`
for each mode, and panel 4 of each dashboard) draw a red dashed line at the
Amber Electric subscription cost for the modelled period, so where the
cumulative-revenue curve crosses it is the break-even time against that fee.

The estimate (`reporting.amber_subscription_estimate`) works as follows:

* **Usage = grid import.** With no site load modelled, battery charging is the
  only import; export does not count as usage.
* **Annualised tier lookup.** The subscription bands are per year, so the
  period import is scaled by `365.25 / period_days` before choosing the tier
  (`≤10,000 kWh/yr → $25/mo`, `≤20,000 → $50`, `≤50,000 → $125`,
  `≤100,000 → $200`, above → `$350`).
* **Flat monthly fee.** Cost = `monthly_fee × calendar months touched` (a
  partial month counts as a full fee, because Amber bills monthly — a calendar
  month is never prorated by days).

The fee is a break-even reference only: it appears in the summary but is **not**
deducted from the benchmark revenue. Annualising a one-month run to a year is
an extrapolation, and the benchmark has no site load or solar generation, so
treat the tier as indicative rather than a billed amount.

## Validation and tests

* `validation.py` independently recomputes the SOC trajectory, energy flows,
  revenue and EFC from the schedule and raises on any violation — it never
  trusts the solver. It also checks the battery-side power against the DC
  rating, and it is basis-aware for the Mode 1 cap. In LP-relaxation mode it
  reports the simultaneous-flow intervals instead of failing, and labels the
  schedule as an upper bound.
* `sanity_tests.py`:

  | Test | Covers |
  |---|---|
  | A | constant price → no artificial arbitrage |
  | B | cheap then expensive → charge low, discharge high |
  | C | negative price → paid to charge |
  | D | SOC limits never breached |
  | E | meter-side power limits never breached |
  | F | efficiency accounting / SOC transition / RTE |
  | G | `equal_initial` vs `free` terminal SOC |
  | H | Mode 1 daily cycle cap respected and actively used |
  | I | revenue equals a hand-computed optimum (exact) |
  | J | MILP blocks the negative-price loophole; MILP ≤ LP bound |
  | K | battery-side current within the C-rate / cable rating both ways |
  | L | every Mode 1 cycle basis binds exactly at the cap |
  | M | data pipeline: interval mismatch, short coverage, duplicate resolution, malformed rows |
  | N | grid connection: phases/current capacity, static export/import caps, validator enforcement |
  | O | Amber subscription: annualised tier boundaries and flat monthly-fee billing |

## Project layout

```
bess_TOS/
├── main.py            orchestration + CLI overrides
├── config.py          all user settings + derived quantities
├── helper.py          reusable NEM data pipeline
├── optimizer.py       (MI)LP model, HiGHS solve, LP bound, schedule frame
├── validation.py      independent constraint/revenue checks
├── reporting.py       performance metrics + summary formatting
├── visualization.py   four-panel + cumulative-revenue plots
├── sanity_tests.py    synthetic tests A–O
├── requirements.txt
├── generated_plot.png       4-panel diagnostic (with break-even line)
├── generated_energy_plot.png cumulative grid energy (kWh)
├── data/              AEMO DISPATCHPRICE archives (+ provenance README)
└── outputs/           per-mode CSV + summary + data manifest + plots
```

## Scope / deliberately excluded constraints

Not requested and therefore **not** implemented, to keep a clean baseline:
minimum charge/discharge duration, ramp-rate limits, flexible/dynamic export
limits and site load or solar generation (static connection, import and export
caps *are* modelled — see "Grid connection limit"), auxiliary/standby
consumption, degradation or cycle-cost penalties, reserve requirements, and
price forecasting. Each can be added as an additional linear constraint
without changing the architecture.

Because none of those are modelled, the benchmark is optimistic in more ways
than perfect foresight alone: with `MIP_REL_GAP=1e-4` the reported revenue is
within 0.01 % of the exact optimum, the schedule may reverse direction between
adjacent 5-minute intervals, and the free-terminal variant can monetise the
initial SOC. All three are visible in the summary (`achieved gap`,
`number of charge/discharge events`, `terminal SOC mode`).

## A note on scope: sizing

The project began as "a TOS approach for **sizing** a BESS". The current
implementation evaluates one fixed pack per run; it does not sweep capacities,
apply capital costs or optimise a size. Sizing studies can be built on top of it
by running several configurations (e.g. a shell loop over
`--capacity-kwh ... --voltage ... --ah ... --outdir ...`) and comparing net
revenue, but a cost/degradation model would have to be added first.

Note also that the shipped dataset is **one month for one region** (VIC1,
August 2026: no price above $406/MWh, ~10.8 % negative intervals). Revenue
extrapolated from a single benign month is not a sizing-grade estimate; extend
`data/` to a full year before drawing capacity conclusions.
