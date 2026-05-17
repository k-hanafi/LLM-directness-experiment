#!/usr/bin/env python3
"""Build column-minimal master CSV for the directness experiment.

Reads the full Khaled-style dataset, keeps only columns used by:
  - src.formatter (user message fields + identity)
  - scripts/compute_fame_proxy.py (fame proxy; not sent to the model)

Coalesces founding into a single ``founded_month_year`` column (``%b %Y``, e.g.
``Nov 2016``): prefer parsing ``founded_date``; if missing/invalid, use numeric
``year_founded`` as January of that year (``Jan YYYY``).

Coalesces long text into ``long_description`` (first non-empty among, in order:
``Long description``, ``long_description``, ``description``).

Usage:
    python scripts/build_master_directness_csv.py
    python scripts/build_master_directness_csv.py --input /path/to/khaled.csv \\
        --output data/master_csv_directness_experiment.csv --chunksize 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "company_us_all_var_Khaled.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "master_csv_directness_experiment.csv"

# Columns required to exist on the input header (Crunchbase-style export).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "org_uuid",
    "name",
    "address",
    "city",
    "state_code",
    "postal_code",
    "short_description",
    "category_list",
    "category_groups_list",
    "rank",
    "total_funding_usd",
    "num_funding_rounds",
    "homepage_url",
    "linkedin_url",
    "twitter_url",
    "facebook_url",
)

# At least one founding column should be present to populate ``founded_month_year``.
FOUNDING_SOURCE_COLUMNS: tuple[str, ...] = ("founded_date", "year_founded")

# Long description: first non-empty wins in this order (matches formatter precedence).
LONG_DESC_SOURCE_COLUMNS: tuple[str, ...] = (
    "Long description",
    "long_description",
    "description",
)

# Written to the master file (stable order).
MASTER_OUTPUT_COLUMNS: tuple[str, ...] = (
    "org_uuid",
    "name",
    "address",
    "city",
    "state_code",
    "postal_code",
    "short_description",
    "long_description",
    "category_list",
    "category_groups_list",
    "founded_month_year",
    "rank",
    "total_funding_usd",
    "num_funding_rounds",
    "homepage_url",
    "linkedin_url",
    "twitter_url",
    "facebook_url",
)


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _validate_header(columns: list[str]) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
    if missing:
        print(f"ERROR: input CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if not any(c in columns for c in FOUNDING_SOURCE_COLUMNS):
        print(
            "ERROR: input CSV must include at least one of: "
            f"{list(FOUNDING_SOURCE_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _usecols(columns: list[str]) -> list[str]:
    """Columns to read from disk (subset of file)."""
    want: set[str] = set(REQUIRED_INPUT_COLUMNS)
    want.update(c for c in FOUNDING_SOURCE_COLUMNS if c in columns)
    want.update(c for c in LONG_DESC_SOURCE_COLUMNS if c in columns)
    return sorted(want)


def _coalesce_long_description(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    out = pd.Series("", index=idx, dtype=object)
    for key in LONG_DESC_SOURCE_COLUMNS:
        if key not in df.columns:
            continue
        s = pd.Series(df[key], dtype="string").fillna("").str.strip()
        s = s.mask(s.str.lower().isin(["", "nan", "none", "nat"]))
        take = (out == "") & (s != "")
        out = out.where(~take, s)
    return out.fillna("")


def _compute_founded_month_year(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    if "founded_date" in df.columns:
        fd = df["founded_date"]
    else:
        fd = pd.Series(pd.NA, index=idx, dtype=object)
    ts_fd = pd.to_datetime(fd, errors="coerce", dayfirst=True)

    if "year_founded" in df.columns:
        yf = pd.to_numeric(df["year_founded"], errors="coerce")
    else:
        yf = pd.Series(np.nan, index=idx, dtype=float)

    y_int = yf.round()
    mask_year = ts_fd.isna() & y_int.notna() & (y_int >= 1900) & (y_int <= 2100)
    ts_y = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    if mask_year.any():
        ts_y.loc[mask_year] = pd.to_datetime(
            y_int.loc[mask_year].astype(int).astype(str) + "-01-01",
            format="%Y-%m-%d",
            errors="coerce",
        )

    combined = ts_fd.combine_first(ts_y)
    out = combined.dt.strftime("%b %Y")
    out = out.where(combined.notna(), "")
    return out.fillna("").replace("nan", "")


def _transform_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in REQUIRED_INPUT_COLUMNS:
        out[col] = df[col]
    out["long_description"] = _coalesce_long_description(df)
    out["founded_month_year"] = _compute_founded_month_year(df)
    # List (not tuple): pandas treats a tuple of labels as one MultiIndex key.
    return out[list(MASTER_OUTPUT_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Master CSV path")
    parser.add_argument("--chunksize", type=int, default=50_000, help="Rows per read chunk")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    all_columns = _read_header(args.input)
    _validate_header(all_columns)
    cols_to_read = _usecols(all_columns)

    dropped = sorted(set(all_columns) - set(cols_to_read))
    long_present = [c for c in LONG_DESC_SOURCE_COLUMNS if c in all_columns]
    founding_present = [c for c in FOUNDING_SOURCE_COLUMNS if c in all_columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    nonempty_founded = 0
    nonempty_long = 0
    first_chunk = True

    reader = pd.read_csv(
        args.input,
        usecols=cols_to_read,
        chunksize=args.chunksize,
        dtype={"postal_code": "string"},
    )
    for chunk in reader:
        transformed = _transform_chunk(chunk)
        transformed.to_csv(
            args.output,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )
        n = len(transformed)
        total_rows += n
        nonempty_founded += int((transformed["founded_month_year"] != "").sum())
        nonempty_long += int((transformed["long_description"] != "").sum())
        first_chunk = False

    pct_f = 100.0 * nonempty_founded / total_rows if total_rows else 0.0
    pct_l = 100.0 * nonempty_long / total_rows if total_rows else 0.0

    print("=" * 60)
    print("Master directness CSV build complete")
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Rows:        {total_rows:,}")
    print(f"  Chunksize:   {args.chunksize:,}")
    print(f"  Long-desc source columns present in input: {long_present or '(none)'}")
    print(f"  Founding source columns present in input: {founding_present}")
    print(f"  Rows with non-empty founded_month_year: {nonempty_founded:,} ({pct_f:.1f}%)")
    print(f"  Rows with non-empty long_description:   {nonempty_long:,} ({pct_l:.1f}%)")
    print(f"  Columns written ({len(MASTER_OUTPUT_COLUMNS)}): {', '.join(MASTER_OUTPUT_COLUMNS)}")
    print(f"  Input columns not carried to master ({len(dropped)}): {', '.join(dropped[:30])}"
          + (" ..." if len(dropped) > 30 else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
