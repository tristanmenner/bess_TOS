"""Reusable NEM DISPATCHPRICE data pipeline.

The functions in this module are intentionally free of any BESS/optimisation
logic so they can be reused by future trading/forecasting projects:

    discover_price_files  ->  read_aemo_mms_file  ->  combine_data
        ->  clean_data  ->  filter_region  ->  filter_date_range
        ->  check_completeness  ->  prepare_price_series

Data format
-----------
AEMO MMS ``PUBLIC_ARCHIVE#DISPATCHPRICE`` files contain a few ``C`` metadata
lines, a single ``I`` header line, ``D`` data rows and a ``C`` trailer.  The
first four comma-separated fields (e.g. ``D,DISPATCH,PRICE,5``) are not column
names; the real columns start at ``SETTLEMENTDATE``.  The fourth field is the
**interval length in minutes**, which is parsed and validated against
``Config.INTERVAL_MINUTES`` so a 30-minute file can never be silently priced as
a 5-minute one.

Timestamps are interval-ENDING and expressed in NEM time (AEST, UTC+10, no
daylight saving). Naive datetimes are therefore correct and no timezone
conversion is performed.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NEM_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"

TIDY_COLUMNS = ["timestamp", "interval_start", "region", "price_mwh", "price_kwh", "day"]

# Column names for the four leading MMS metadata fields that are not part of the
# published table schema.
LEADING_MMS_FIELDS = ["RECORD_TYPE", "DISPATCH_TYPE", "TABLE_NAME", "MMS_INTERVAL_MINUTES"]

_PRICE_BLANK = {"", "nan", "NaN", "none", "None", "null", "NULL", "NA", "N/A"}


class DataValidationError(RuntimeError):
    """Raised when the input data cannot support a trustworthy optimisation."""


@dataclass
class DataReport:
    """Structured result of a data validation pass."""

    files: list[str] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    regions: list[str] = field(default_factory=list)
    n_rows_raw: int = 0
    n_rows_clean: int = 0
    n_duplicate_timestamps: int = 0
    n_missing_price: int = 0
    n_invalid_price: int = 0
    n_unparseable_timestamps: int = 0
    n_missing_intervals: int = 0
    n_expected_intervals: int = 0
    n_covered_intervals: int = 0
    n_missing_head: int = 0
    n_missing_tail: int = 0
    mms_interval_minutes: float | None = None
    missing_intervals: list[pd.Timestamp] = field(default_factory=list)
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def coverage_complete(self) -> bool:
        return self.n_missing_intervals == 0 and self.n_expected_intervals > 0

    def summary_lines(self) -> list[str]:
        lines = [
            f"files discovered          : {len(self.files)}",
            f"rows read (raw)           : {self.n_rows_raw}",
            f"rows after cleaning       : {self.n_rows_clean}",
            f"regions available         : {', '.join(self.regions) if self.regions else '(none)'}",
            f"unparseable timestamps    : {self.n_unparseable_timestamps}",
            f"duplicate timestamps      : {self.n_duplicate_timestamps}",
            f"missing prices            : {self.n_missing_price}",
            f"invalid (non-numeric)     : {self.n_invalid_price}",
        ]
        if self.n_expected_intervals:
            lines.append(
                f"expected intervals        : {self.n_expected_intervals:,}"
            )
            lines.append(
                f"covered intervals         : {self.n_covered_intervals:,}"
                f"  ({100.0 * self.n_covered_intervals / self.n_expected_intervals:.4f}%)"
            )
            lines.append(
                f"missing 5-min intervals   : {self.n_missing_intervals:,}"
                f"  (head {self.n_missing_head:,}, tail {self.n_missing_tail:,})"
            )
        if self.first_timestamp is not None and self.last_timestamp is not None:
            lines.append(
                f"coverage                  : {self.first_timestamp} -> {self.last_timestamp}"
            )
        if self.mms_interval_minutes is not None:
            lines.append(f"MMS header interval       : {self.mms_interval_minutes:g} min")
        for name, digest in self.file_hashes.items():
            lines.append(f"sha256({name}) : {digest}")
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.extend(f"ERROR  : {e}" for e in self.errors)
        return lines


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_price_files(data_dir: Path | str) -> list[Path]:
    """Return every CSV/CSV-like file under ``data_dir`` in a stable order.

    No filename is hard-coded; both ``.csv`` and the AEMO archive naming
    convention are picked up.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataValidationError(f"Data directory does not exist: {data_dir}")
    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".txt"}
    )
    if not files:
        raise DataValidationError(f"No CSV files found in {data_dir}")
    return files


# --------------------------------------------------------------------------- #
# Low-level reader
# --------------------------------------------------------------------------- #
def read_aemo_mms_file(path: Path | str) -> pd.DataFrame:
    """Parse a single AEMO MMS CSV file into a DataFrame of strings.

    Handles the ``C``/``I``/``D`` record structure and the four leading
    non-column fields.  Every ``D`` row is required to have exactly the same
    number of fields as the ``I`` header, so a format change raises instead of
    silently shifting columns.  The parsed MMS interval length (minutes) is
    exposed through ``DataFrame.attrs["mms_interval_minutes"]``.
    """
    path = Path(path)
    text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 always succeeds
        raise DataValidationError(f"Could not decode {path}")

    lines = text.splitlines()
    header_idx = None
    header_fields: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("I,"):
            header_idx = i
            header_fields = next(csv.reader([line]))
            break
    if header_idx is None:
        raise DataValidationError(f"No 'I' header record found in {path}")

    names = header_fields[4:]
    if not names:
        raise DataValidationError(f"Header record in {path} has no column names")
    expected_fields = len(header_fields)

    data_lines = [
        (lineno, ln)
        for lineno, ln in enumerate(lines[header_idx + 1 :], start=header_idx + 2)
        if ln.startswith("D,")
    ]
    if not data_lines:
        raise DataValidationError(f"No 'D' data records found in {path}")

    rows: list[list[str]] = []
    for lineno, line in data_lines:
        fields = next(csv.reader([line]))
        if len(fields) != expected_fields:
            raise DataValidationError(
                f"{path.name}: line {lineno} has {len(fields)} fields but the "
                f"'I' header declares {expected_fields}; refusing to guess column "
                "alignment."
            )
        rows.append(fields)

    df = pd.DataFrame(rows, columns=LEADING_MMS_FIELDS + names)

    # Interval length from the header (e.g. the '5' in 'D,DISPATCH,PRICE,5').
    interval: float | None = None
    try:
        interval = float(header_fields[3])
    except (TypeError, ValueError):
        interval = None

    df = df.drop(columns=["RECORD_TYPE", "DISPATCH_TYPE", "TABLE_NAME", "MMS_INTERVAL_MINUTES"])
    df = df.reset_index(drop=True)
    df["__source_file"] = path.name
    df.attrs["mms_interval_minutes"] = interval
    return df


def combine_data(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate monthly/regional frames into one long frame."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise DataValidationError("No data frames to combine")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["mms_interval_minutes"] = frames[0].attrs.get("mms_interval_minutes")
    return combined


# --------------------------------------------------------------------------- #
# Cleaning / validation
# --------------------------------------------------------------------------- #
def clean_data(
    df: pd.DataFrame,
    price_column: str = "RRP",
    timestamp_column: str = "SETTLEMENTDATE",
    region_column: str = "REGIONID",
    report: DataReport | None = None,
) -> pd.DataFrame:
    """Coerce types, drop duplicates and sort chronologically.

    Rows with unparseable timestamps or invalid prices are removed and counted
    (never silently imputed).

    Duplicate ``(timestamp, region)`` records (e.g. re-runs / intervention runs
    inside one archive) are resolved by preferring, in order: the highest
    ``INTERVENTION`` flag, the highest ``RUNNO``, then the last record in file
    order.  When those MMS columns are absent, the last record wins.
    """
    report = report or DataReport()
    for col in (timestamp_column, region_column, price_column):
        if col not in df.columns:
            raise DataValidationError(f"Required column '{col}' missing from input")

    out = df.copy()
    out["__row_order"] = np.arange(len(out))

    out[timestamp_column] = pd.to_datetime(
        out[timestamp_column].astype(str).str.strip(),
        format=NEM_TIMESTAMP_FORMAT,
        errors="coerce",
    )
    bad_ts = int(out[timestamp_column].isna().sum())
    report.n_unparseable_timestamps += bad_ts

    raw_price = out[price_column]
    blank_price = raw_price.isna() | raw_price.astype(str).str.strip().isin(_PRICE_BLANK)
    out[price_column] = pd.to_numeric(raw_price, errors="coerce")
    report.n_missing_price += int(blank_price.sum())
    report.n_invalid_price += int((out[price_column].isna() & ~blank_price).sum())

    out[region_column] = out[region_column].astype(str).str.strip()

    out = out.dropna(subset=[timestamp_column, price_column])
    out = out[out[region_column] != ""]

    # ---- duplicate resolution (latest run wins) ---------------------------
    sort_cols = [timestamp_column, region_column]
    if "INTERVENTION" in out.columns:
        out["__intervention"] = pd.to_numeric(out["INTERVENTION"], errors="coerce").fillna(0).astype("int64")
        sort_cols.append("__intervention")
    if "RUNNO" in out.columns:
        out["__runno"] = pd.to_numeric(out["RUNNO"], errors="coerce").fillna(-1).astype("int64")
        sort_cols.append("__runno")
    sort_cols.append("__row_order")

    before = len(out)
    out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates(subset=[timestamp_column, region_column], keep="last")
    report.n_duplicate_timestamps += before - len(out)

    drop_cols = [c for c in ("__row_order", "__intervention", "__runno") if c in out.columns]
    out = out.drop(columns=drop_cols)
    out = out.sort_values([timestamp_column, region_column]).reset_index(drop=True)
    report.n_rows_clean = len(out)
    return out


def filter_region(df: pd.DataFrame, region: str, region_column: str = "REGIONID") -> pd.DataFrame:
    """Keep only the requested NEM region."""
    if region_column not in df.columns:
        raise DataValidationError(f"Column '{region_column}' missing")
    out = df[df[region_column].astype(str).str.strip() == region].copy()
    if out.empty:
        available = sorted(df[region_column].astype(str).unique().tolist())
        raise DataValidationError(
            f"Region {region!r} not found ({len(df)} rows after cleaning); "
            f"available: {available or '(none)'}"
        )
    return out.reset_index(drop=True)


def filter_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    timestamp_column: str = "SETTLEMENTDATE",
    interval_minutes: float = 5.0,
) -> pd.DataFrame:
    """Inclusive date-range filter based on the interval-START timestamp.

    AEMO timestamps are interval-ending, so the interval ending at 00:00 on the
    start date belongs to the *previous* day and must be excluded. Using
    ``interval_start = timestamp - interval`` gives the intuitive, inclusive
    behaviour and correctly attributes the midnight boundary.
    """
    interval = pd.to_timedelta(interval_minutes, unit="m")
    interval_start = df[timestamp_column] - interval
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (interval_start >= start) & (interval_start < end)
    out = df.loc[mask].copy()
    if out.empty:
        raise DataValidationError(
            f"No data between {start_date} and {end_date} (inclusive)"
        )
    return out.reset_index(drop=True)


def check_completeness(
    df: pd.DataFrame,
    config,
    report: DataReport | None = None,
    timestamp_column: str = "SETTLEMENTDATE",
) -> DataReport:
    """Validate coverage against the *requested* period, not just present rows.

    The expected series is every ``Config.INTERVAL_MINUTES`` interval-start in
    ``[START_DATE, END_DATE + 1 day)``; the present series is derived from the
    interval-ending timestamps.  Missing intervals are counted overall and split
    into head (before the first observation) and tail (after the last), so a
    truncated archive cannot masquerade as a complete period.

    With ``Config.REQUIRE_FULL_COVERAGE`` (default) any gap raises
    :class:`DataValidationError`; otherwise it is recorded as a warning.
    """
    report = report or DataReport()
    interval = pd.Timedelta(minutes=config.INTERVAL_MINUTES)
    interval_start = (df[timestamp_column] - interval).sort_values()
    present = pd.DatetimeIndex(interval_start.unique())

    expected = pd.date_range(
        pd.Timestamp(config.START_DATE),
        pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1),
        freq=interval,
        inclusive="left",
    )
    missing = expected.difference(present)
    report.n_expected_intervals = len(expected)
    report.n_covered_intervals = len(expected.intersection(present))
    report.n_missing_intervals = len(missing)
    if len(present):
        report.n_missing_head = int((expected < present.min()).sum())
        report.n_missing_tail = int((expected > present.max()).sum())
    else:  # pragma: no cover - filter_date_range already rejects empty input
        report.n_missing_head = len(expected)
        report.n_missing_tail = 0
    report.missing_intervals = list(missing[:20])

    if len(missing):
        msg = (
            f"{len(missing):,} of {len(expected):,} expected "
            f"{config.INTERVAL_MINUTES:g}-min intervals in "
            f"[{config.START_DATE}, {config.END_DATE}] are missing "
            f"(head {report.n_missing_head:,}, tail {report.n_missing_tail:,}; "
            f"first few: {[str(m) for m in missing[:5]]})."
        )
        if config.REQUIRE_FULL_COVERAGE:
            report.add_error(msg)
            raise DataValidationError(msg + " Prices are NOT imputed.")
        report.add_warning(msg + " Prices are NOT imputed.")
    return report


# --------------------------------------------------------------------------- #
# Tidy price-series construction
# --------------------------------------------------------------------------- #
def prepare_price_series(config, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, DataReport]:
    """Build the tidy price series used by the optimiser.

    Returns ``(tidy_df, report)`` where ``tidy_df`` has columns
    ``timestamp, interval_start, region, price_mwh, price_kwh, day`` in
    chronological order.  ``day`` follows ``Config.DAY_BASIS`` (calendar day by
    default, or the 04:00-to-04:00 NEM trading day).
    """
    report = DataReport()
    if df is None:
        frames = []
        files = discover_price_files(config.DATA_DIR)
        report.files = [p.name for p in files]
        report.file_hashes = {p.name: _sha256(p) for p in files}
        for path in files:
            frames.append(read_aemo_mms_file(path))
        df = combine_data(frames)

    mms_interval = df.attrs.get("mms_interval_minutes")
    if mms_interval is not None:
        report.mms_interval_minutes = float(mms_interval)
        if abs(float(mms_interval) - float(config.INTERVAL_MINUTES)) > 1e-9:
            raise DataValidationError(
                f"MMS header declares a {float(mms_interval):g}-minute interval but "
                f"INTERVAL_MINUTES={config.INTERVAL_MINUTES:g}; refusing to price the "
                "file with the wrong interval length."
            )

    report.n_rows_raw = len(df)
    report.regions = sorted(df["REGIONID"].astype(str).str.strip().unique().tolist())

    df = clean_data(
        df,
        price_column=config.PRICE_COLUMN,
        region_column="REGIONID",
        report=report,
    )
    df = filter_region(df, config.REGION)
    df = filter_date_range(
        df,
        config.START_DATE,
        config.END_DATE,
        interval_minutes=config.INTERVAL_MINUTES,
    )
    report = check_completeness(df, config, report=report)

    interval = pd.to_timedelta(config.INTERVAL_MINUTES, unit="m")
    tidy = pd.DataFrame(
        {
            "timestamp": df["SETTLEMENTDATE"].values,
            "interval_start": (df["SETTLEMENTDATE"] - interval).values,
            "region": config.REGION,
            "price_mwh": pd.to_numeric(df[config.PRICE_COLUMN]).values,
        }
    )
    tidy["price_kwh"] = tidy["price_mwh"] * config.price_scale
    interval_start = pd.to_datetime(tidy["interval_start"])
    if config.DAY_BASIS == "nem_trading":
        # AEMO trading day: 04:05 (interval ending) .. 04:00 (interval ending),
        # i.e. interval-starts from 04:00 to 03:55 the next day.
        tidy["day"] = (interval_start - pd.Timedelta(hours=4)).dt.normalize()
    else:
        tidy["day"] = interval_start.dt.normalize()
    tidy = tidy[TIDY_COLUMNS].sort_values("interval_start").reset_index(drop=True)

    if int(tidy["price_kwh"].isna().sum()):
        report.add_error("Missing prices remain after cleaning")
        raise DataValidationError("Missing prices in the prepared series")
    if len(tidy) < 1:
        raise DataValidationError("Prepared price series is empty")

    report.first_timestamp = tidy["timestamp"].min()
    report.last_timestamp = tidy["timestamp"].max()
    return tidy, report


__all__ = [
    "DataValidationError",
    "DataReport",
    "NEM_TIMESTAMP_FORMAT",
    "TIDY_COLUMNS",
    "discover_price_files",
    "read_aemo_mms_file",
    "combine_data",
    "clean_data",
    "filter_region",
    "filter_date_range",
    "check_completeness",
    "prepare_price_series",
]
