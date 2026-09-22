"""Reusable NEM DISPATCHPRICE data pipeline.

The functions in this module are intentionally free of any BESS/optimisation
logic so they can be reused by future trading/forecasting projects:

    discover_price_files  ->  read_aemo_mms_file  ->  combine_data
        ->  clean_data  ->  filter_region  ->  filter_date_range
        ->  prepare_price_series

Data format
-----------
AEMO MMS ``PUBLIC_ARCHIVE#DISPATCHPRICE`` files contain a few ``C`` metadata
lines, a single ``I`` header line, ``D`` data rows and a ``C`` trailer. The
first four comma-separated fields (e.g. ``D,DISPATCH,PRICE,5``) are not column
names; the real columns start at ``SETTLEMENTDATE``.

Timestamps are interval-ENDING and expressed in NEM time (AEST, UTC+10, no
daylight saving). Naive datetimes are therefore correct and no timezone
conversion is performed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

NEM_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"

TIDY_COLUMNS = ["timestamp", "interval_start", "region", "price_mwh", "price_kwh", "day"]


class DataValidationError(RuntimeError):
    """Raised when the input data cannot support a trustworthy optimisation."""


@dataclass
class DataReport:
    """Structured result of a data validation pass."""

    files: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    n_rows_raw: int = 0
    n_rows_clean: int = 0
    n_duplicate_timestamps: int = 0
    n_missing_price: int = 0
    n_invalid_price: int = 0
    n_unparseable_timestamps: int = 0
    n_missing_intervals: int = 0
    missing_intervals: list[pd.Timestamp] = field(default_factory=list)
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

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
            f"missing 5-min intervals   : {self.n_missing_intervals}",
        ]
        if self.first_timestamp is not None and self.last_timestamp is not None:
            lines.append(
                f"coverage                  : {self.first_timestamp} -> {self.last_timestamp}"
            )
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.extend(f"ERROR  : {e}" for e in self.errors)
        return lines


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
    non-column fields. The returned frame keeps the original MMS column names
    (``SETTLEMENTDATE``, ``REGIONID``, ``RRP``, ...).
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

    data_lines = [ln for ln in lines[header_idx + 1 :] if ln.startswith("D,")]
    if not data_lines:
        raise DataValidationError(f"No 'D' data records found in {path}")

    # csv handles the quoted timestamp fields; declare columns explicitly.
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        names=names,
        header=None,
        dtype=str,
        skipinitialspace=True,
    )
    df["__source_file"] = path.name
    return df


def combine_data(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate monthly/regional frames into one long frame."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise DataValidationError("No data frames to combine")
    return pd.concat(frames, ignore_index=True, sort=False)


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
    (never silently imputed). Duplicates are resolved by keeping the first
    record per (timestamp, region); in MMS archives a later intervention run
    would otherwise duplicate the interval.
    """
    report = report or DataReport()
    for col in (timestamp_column, region_column, price_column):
        if col not in df.columns:
            raise DataValidationError(f"Required column '{col}' missing from input")

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(
        out[timestamp_column].astype(str).str.strip(),
        format=NEM_TIMESTAMP_FORMAT,
        errors="coerce",
    )
    bad_ts = int(out[timestamp_column].isna().sum())
    report.n_unparseable_timestamps += bad_ts

    out[price_column] = pd.to_numeric(out[price_column], errors="coerce")
    bad_price = int(out[price_column].isna().sum())
    report.n_invalid_price += bad_price

    out[region_column] = out[region_column].astype(str).str.strip()

    out = out.dropna(subset=[timestamp_column, price_column])
    out = out[out[region_column] != ""]

    before = len(out)
    out = out.sort_values([region_column, timestamp_column])
    out = out.drop_duplicates(subset=[timestamp_column, region_column], keep="first")
    report.n_duplicate_timestamps += before - len(out)

    out = out.sort_values([timestamp_column, region_column]).reset_index(drop=True)
    report.n_rows_clean = len(out)
    return out


def validate_data(
    df: pd.DataFrame,
    region: str,
    start_date: str,
    end_date: str,
    interval_minutes: float = 5.0,
    price_column: str = "RRP",
    timestamp_column: str = "SETTLEMENTDATE",
    region_column: str = "REGIONID",
    report: DataReport | None = None,
) -> DataReport:
    """Validate the cleaned frame against the requested slice.

    Fatal problems raise :class:`DataValidationError`; recoverable issues are
    recorded as warnings so the caller can surface them without guessing.
    """
    report = report or DataReport()
    report.regions = sorted(df[region_column].unique().tolist())
    if region not in report.regions:
        report.add_error(f"Region {region!r} not present (available: {report.regions})")
        raise DataValidationError(f"Region {region!r} not present: {report.regions}")

    sub = df[df[region_column] == region]
    if sub.empty:
        raise DataValidationError(f"No rows for region {region!r}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise DataValidationError("START_DATE must not be after END_DATE")
    report.first_timestamp = sub[timestamp_column].min()
    report.last_timestamp = sub[timestamp_column].max()

    return report


def filter_region(df: pd.DataFrame, region: str, region_column: str = "REGIONID") -> pd.DataFrame:
    """Keep only the requested NEM region."""
    if region_column not in df.columns:
        raise DataValidationError(f"Column '{region_column}' missing")
    out = df[df[region_column].astype(str).str.strip() == region].copy()
    if out.empty:
        available = sorted(df[region_column].astype(str).unique().tolist())
        raise DataValidationError(f"Region {region!r} not found; available: {available}")
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
    interval_minutes: float = 5.0,
    timestamp_column: str = "SETTLEMENTDATE",
    report: DataReport | None = None,
) -> DataReport:
    """Report missing intervals between the first and last observation."""
    report = report or DataReport()
    interval = pd.to_timedelta(interval_minutes, unit="m")
    interval_start = (df[timestamp_column] - interval).sort_values()
    expected = pd.date_range(interval_start.min(), interval_start.max(), freq=interval)
    present = pd.DatetimeIndex(interval_start.unique())
    missing = expected.difference(present)
    report.n_missing_intervals = len(missing)
    report.missing_intervals = list(missing[:20])
    if len(missing):
        report.add_warning(
            f"{len(missing)} expected {interval_minutes:g}-min intervals are missing "
            f"(first few: {[str(m) for m in missing[:5]]}); prices are NOT imputed."
        )
    return report


# --------------------------------------------------------------------------- #
# Tidy price-series construction
# --------------------------------------------------------------------------- #
def prepare_price_series(config, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, DataReport]:
    """Build the tidy 5-minute price series used by the optimiser.

    Returns ``(tidy_df, report)`` where ``tidy_df`` has columns
    ``timestamp, interval_start, region, price_mwh, price_kwh, day`` in
    chronological order. ``day`` (the interval-start date) is used for daily
    aggregation.
    """
    report = DataReport()
    if df is None:
        frames = []
        files = discover_price_files(config.DATA_DIR)
        report.files = [p.name for p in files]
        for path in files:
            frames.append(read_aemo_mms_file(path))
        df = combine_data(frames)
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
    report = check_completeness(
        df, interval_minutes=config.INTERVAL_MINUTES, report=report
    )

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
    tidy["day"] = tidy["interval_start"].dt.normalize()
    tidy = tidy[TIDY_COLUMNS].sort_values("interval_start").reset_index(drop=True)

    report.n_missing_price = int(tidy["price_kwh"].isna().sum())
    if report.n_missing_price:
        report.add_error("Missing prices remain after cleaning")
        raise DataValidationError("Missing prices in the prepared series")
    if len(tidy) < 1:
        raise DataValidationError("Prepared price series is empty")
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
    "validate_data",
    "filter_region",
    "filter_date_range",
    "check_completeness",
    "prepare_price_series",
]
