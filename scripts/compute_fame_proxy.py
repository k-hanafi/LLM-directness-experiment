"""Build the composite fame score and quartile assignments for every org_uuid.

Pipeline:
  1. Direction-align: invert rank so higher = more famous.
  2. Tame heavy tails: log-transform funding, rounds, inverted-rank.
  3. Z-score every continuous column to a common scale (median imputation).
  4. Combine equal-weighted continuous z-scores + half-weighted URL flags.
  5. Bin the composite into quartiles Q1 (most obscure) -> Q4 (most famous).

Output:
  outputs/analysis/fame_quartiles.csv          (org_uuid, fame_score, fame_quartile)
  outputs/analysis/fame_diagnostics.json       (face-validity sample, weights, scaler params)

Run once after scripts/import_baseline.py and before
scripts/analyze_directness.py.

Usage:
    python scripts/compute_fame_proxy.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "company_us_all_var_Khaled.csv"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
QUARTILES_OUT = ANALYSIS_DIR / "fame_quartiles.csv"
DIAGNOSTICS_OUT = ANALYSIS_DIR / "fame_diagnostics.json"

URL_COLUMNS = ["homepage_url", "linkedin_url", "twitter_url", "facebook_url"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _safe_log10(x: pd.Series) -> pd.Series:
    """log10(x + 1) with NaN passthrough; coerces non-numeric to NaN."""
    numeric = pd.to_numeric(x, errors="coerce")
    return np.log10(numeric.clip(lower=0) + 1)


def _zscore(x: pd.Series) -> pd.Series:
    """Median-impute then z-score. Returns NaN-free series."""
    median = x.median(skipna=True)
    filled = x.fillna(median)
    std = filled.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=x.index)
    return (filled - filled.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"Input CSV (default: {DEFAULT_DATA})")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    needed = ["org_uuid", "name", "rank", "total_funding_usd", "num_funding_rounds"] + URL_COLUMNS
    df = pd.read_csv(args.data, usecols=lambda c: c in needed or c == "org_uuid")

    rank_numeric = pd.to_numeric(df["rank"], errors="coerce")
    max_rank = rank_numeric.max(skipna=True)
    rank_inv = max_rank - rank_numeric + 1
    log_rank_inv = _safe_log10(rank_inv)
    log_funding = _safe_log10(df["total_funding_usd"])
    log_rounds = _safe_log10(df["num_funding_rounds"])

    z_log_rank_inv = _zscore(log_rank_inv)
    z_log_funding = _zscore(log_funding)
    z_log_rounds = _zscore(log_rounds)

    url_present = pd.DataFrame({
        col: df[col].notna() & (df[col].astype(str).str.strip() != "")
        for col in URL_COLUMNS
    }).astype(float).sum(axis=1)

    fame_score = (
        z_log_rank_inv
        + z_log_funding
        + z_log_rounds
        + 0.5 * (url_present - url_present.mean()) / max(url_present.std(), 1e-9)
    )

    quartile = pd.qcut(fame_score, q=4, labels=QUARTILE_LABELS)

    out = pd.DataFrame({
        "org_uuid": df["org_uuid"],
        "fame_score": fame_score.round(4),
        "fame_quartile": quartile.astype(str),
    })
    out.to_csv(QUARTILES_OUT, index=False)
    print(f"Wrote {QUARTILES_OUT}  ({len(out):,} rows)")

    sample_top = (
        df.assign(fame_score=fame_score)
          .nlargest(20, "fame_score")[["name", "rank", "total_funding_usd"]]
          .to_dict(orient="records")
    )
    sample_bot = (
        df.assign(fame_score=fame_score)
          .nsmallest(20, "fame_score")[["name", "rank", "total_funding_usd"]]
          .to_dict(orient="records")
    )

    diagnostics = {
        "n_rows": int(len(df)),
        "n_per_quartile": {q: int((quartile == q).sum()) for q in QUARTILE_LABELS},
        "fame_score_summary": {
            "mean": float(fame_score.mean()),
            "std": float(fame_score.std()),
            "min": float(fame_score.min()),
            "max": float(fame_score.max()),
            "p25": float(fame_score.quantile(0.25)),
            "p50": float(fame_score.quantile(0.50)),
            "p75": float(fame_score.quantile(0.75)),
        },
        "weights": {
            "z_log_rank_inv": 1.0,
            "z_log_funding": 1.0,
            "z_log_rounds": 1.0,
            "url_presence_zscaled": 0.5,
        },
        "face_validity_top20": sample_top,
        "face_validity_bottom20": sample_bot,
    }
    DIAGNOSTICS_OUT.write_text(json.dumps(diagnostics, indent=2, default=str))
    print(f"Wrote {DIAGNOSTICS_OUT}")
    print()
    print("Top 5 most-famous (Q4) companies in dataset:")
    for r in sample_top[:5]:
        print(f"  {r['name']!s:60s}  rank={r['rank']!s:>10s}  funding={r['total_funding_usd']!s}")
    print()
    print("Top 5 most-obscure (Q1) companies in dataset:")
    for r in sample_bot[:5]:
        print(f"  {r['name']!s:60s}  rank={r['rank']!s:>10s}  funding={r['total_funding_usd']!s}")


if __name__ == "__main__":
    main()
