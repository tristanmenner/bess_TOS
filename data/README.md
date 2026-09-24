# Input data

The benchmark reads AEMO MMS `DISPATCHPRICE` monthly archives from this
directory. Archive files themselves are **git-ignored** (a monthly file is
~17 MB); this file records what the shipped outputs were produced from.

## Shipped run

| Field | Value |
|---|---|
| File | `PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202608010000.CSV` |
| Source | AEMO MMS `DISPATCHPRICE`, monthly archive (C-record: `SETP.WORLD,DVD_DISPATCHPRICE,AEMO,PUBLIC`, generated `2026/09/08 13:56:34`) |
| Coverage | 2026-08-01 00:05 → 2026-09-01 00:00 (interval-ending, NEM time UTC+10) |
| Rows | 44 640 = 5 regions (NSW1, QLD1, SA1, TAS1, VIC1) x 8 928 five-minute intervals |
| SHA-256 | `59a1adad375ea564a78541d6f7faf710c5cef86cc7a4b3dd2dbd8a6eb95abef4` |

Every run re-computes the SHA-256 of the files it reads, prints it in the
summary's DATA section, and writes `outputs/data_manifest.sha256`. Compare that
manifest against this table before trusting a set of outputs.

If you supply a different archive, the pipeline will:

* reject an MMS header whose interval length disagrees with
  `Config.INTERVAL_MINUTES` (so a 30-minute file can never be priced as
  5-minute);
* reject any `D` row whose field count differs from the `I` header;
* fail (by default) if the requested `START_DATE`..`END_DATE` window is not
  fully covered — set `REQUIRE_FULL_COVERAGE=False` / `--incomplete-data` to
  accept partial data, in which case the gaps are reported in the summary and
  every total is for the covered intervals only.
