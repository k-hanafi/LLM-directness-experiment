"""Compute the directness audit's headline statistical evidence.

All three classification CSVs are produced INTERNALLY by this repo's pipeline
(no import from a sibling repo) so the only experimentally-varied factor is
which fields are populated in the user message.

Inputs:
    outputs/baseline/classified_baseline.csv          (full inputs)
    outputs/arm_a/classified_arm_a.csv                (real name + address)
    outputs/arm_b/classified_arm_b.csv                (anonymized + address)
    outputs/analysis/fame_quartiles.csv               (Q1..Q4 per org_uuid)

Outputs in outputs/analysis/:
    directness_metrics.json                           top-level summary
    confusion_<pair>__<axis>.csv
    kappa_by_fame_quartile.csv                        long-format
    stratified_metrics.csv                            long-format (all strata)
    fallback_rates.csv                                per arm per stratum

Statistical methodology:
    * Raw agreement rate (paired %)
    * Cohen's kappa (sklearn.metrics.cohen_kappa_score)
    * McNemar's test for binary axes (statsmodels)
    * Stuart-Maxwell test for multi-class axes (statsmodels SquareTable.homogeneity)

Pairs analyzed:
    Baseline vs Arm A    (input-stripping effect with real name kept)
    Baseline vs Arm B    (input-stripping effect with name anonymized too)
    Arm A vs Arm B       (pure name-identity effect)

Usage:
    python scripts/analyze_directness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.contingency_tables import SquareTable, mcnemar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

BASELINE_PATH = PROJECT_ROOT / "outputs" / "baseline" / "classified_baseline.csv"
ARM_A_PATH = PROJECT_ROOT / "outputs" / "arm_a" / "classified_arm_a.csv"
ARM_B_PATH = PROJECT_ROOT / "outputs" / "arm_b" / "classified_arm_b.csv"
FAME_PATH = ANALYSIS_DIR / "fame_quartiles.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "master_csv_directness_experiment.csv"

AXES = {
    "ai_native": {"labels": [0, 1], "type": "binary"},
    "subclass": {
        "labels": ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C", "0"],
        "type": "multi",
    },
    "rad_score": {"labels": ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"], "type": "multi"},
    "cohort": {"labels": ["PRE-GENAI", "GENAI-ERA"], "type": "binary"},
}

PAIRS = [
    ("baseline", "arm_a"),
    ("baseline", "arm_b"),
    ("arm_a", "arm_b"),
]


def _load_one(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: missing {path}. Run the corresponding arm first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df = df.rename(columns={"CompanyID": "org_uuid"})
    keep = ["org_uuid"] + list(AXES.keys()) + ["conf_classification", "reasons_3_points"]
    df = df[[c for c in keep if c in df.columns]]
    rename = {c: f"{c}__{suffix}" for c in df.columns if c != "org_uuid"}
    return df.rename(columns=rename)


def load_joined() -> pd.DataFrame:
    """Inner-join the three classification CSVs on org_uuid plus fame quartile."""
    base = _load_one(BASELINE_PATH, "baseline")
    arm_a = _load_one(ARM_A_PATH, "arm_a")
    arm_b = _load_one(ARM_B_PATH, "arm_b")

    df = base.merge(arm_a, on="org_uuid").merge(arm_b, on="org_uuid")

    if FAME_PATH.exists():
        fame = pd.read_csv(FAME_PATH)[["org_uuid", "fame_quartile"]]
        df = df.merge(fame, on="org_uuid", how="left")
    else:
        print(f"WARNING: {FAME_PATH} missing; fame strata will be skipped", file=sys.stderr)
        df["fame_quartile"] = "ALL"

    return df


def _paired_metrics(y1: pd.Series, y2: pd.Series, axis: str) -> dict:
    """Return agreement rate, kappa, and directional test for one (axis, pair)."""
    mask = y1.notna() & y2.notna()
    y1 = y1[mask]
    y2 = y2[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}

    agree = float((y1 == y2).mean())

    info = AXES[axis]
    labels = info["labels"]
    try:
        if info["type"] == "multi":
            kappa = float(cohen_kappa_score(y1, y2, labels=labels))
        else:
            kappa = float(cohen_kappa_score(y1, y2))
    except Exception:
        kappa = float("nan")

    test_stat: float | None = None
    p_value: float | None = None
    test_name: str | None = None

    if info["type"] == "binary":
        cm = confusion_matrix(y1, y2, labels=labels)
        if cm.shape == (2, 2):
            try:
                exact = int((cm[0, 1] + cm[1, 0])) < 25
                res = mcnemar(cm, exact=exact, correction=not exact)
                test_stat = float(res.statistic) if res.statistic is not None else None
                p_value = float(res.pvalue)
                test_name = "mcnemar"
            except Exception:
                pass
    else:
        try:
            cm = confusion_matrix(y1, y2, labels=labels)
            res = SquareTable(cm, shift_zeros=True).homogeneity()
            test_stat = float(res.statistic)
            p_value = float(res.pvalue)
            test_name = "stuart_maxwell"
        except Exception:
            pass

    return {
        "n": n,
        "agreement": round(agree, 4),
        "kappa": round(kappa, 4) if not np.isnan(kappa) else None,
        "test_name": test_name,
        "test_statistic": round(test_stat, 4) if test_stat is not None else None,
        "p_value": float(f"{p_value:.4g}") if p_value is not None else None,
    }


def _confusion_csv(y1: pd.Series, y2: pd.Series, axis: str, name: str) -> Path:
    info = AXES[axis]
    labels = info["labels"]
    mask = y1.notna() & y2.notna()
    cm = confusion_matrix(y1[mask], y2[mask], labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "row=src1"
    df.columns.name = "col=src2"
    out = ANALYSIS_DIR / f"confusion_{name}__{axis}.csv"
    df.to_csv(out)
    return out


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def _data_caveat() -> dict | None:
    if not MASTER_PATH.exists():
        return None
    col = pd.read_csv(MASTER_PATH, usecols=["long_description"])
    nonempty = col["long_description"].fillna("").astype(str).str.strip()
    nonempty = nonempty.mask(nonempty.str.lower().isin(["", "nan", "none", "nat"]))
    pct = round(100.0 * (nonempty != "").mean(), 2)
    return {"long_description_nonempty_pct": pct}


def _build_coverage(n_joined: int) -> dict:
    return {
        "n_master": _count_csv_rows(MASTER_PATH),
        "n_baseline": _count_csv_rows(BASELINE_PATH),
        "n_arm_a": _count_csv_rows(ARM_A_PATH),
        "n_arm_b": _count_csv_rows(ARM_B_PATH),
        "n_joined": n_joined,
    }


def _build_headlines(global_metrics: dict, strata_rows: list[dict]) -> dict:
    ba = global_metrics.get("ai_native", {}).get("baseline__vs__arm_a", {})
    fame_ab = {
        r["stratum_value"]: r
        for r in strata_rows
        if r.get("stratum") == "fame_quartile"
        and r.get("axis") == "ai_native"
        and r.get("pair") == "arm_a__vs__arm_b"
    }
    q1 = fame_ab.get("Q1", {})
    q4 = fame_ab.get("Q4", {})
    kappa_q1 = q1.get("kappa")
    kappa_q4 = q4.get("kappa")
    delta: float | None = None
    if kappa_q1 is not None and kappa_q4 is not None:
        delta = round(float(kappa_q4) - float(kappa_q1), 4)
    return {
        "ai_native_baseline_vs_arm_a_global": {
            "kappa": ba.get("kappa"),
            "agreement": ba.get("agreement"),
            "n": ba.get("n"),
        },
        "ai_native_arm_a_vs_arm_b_fame": {
            "Q1": {
                "kappa": q1.get("kappa"),
                "agreement": q1.get("agreement"),
                "n": q1.get("n"),
            },
            "Q4": {
                "kappa": q4.get("kappa"),
                "agreement": q4.get("agreement"),
                "n": q4.get("n"),
            },
            "kappa_delta_q4_minus_q1": delta,
        },
    }


def _fallback_rate(df: pd.DataFrame, suffix: str) -> float:
    """Fraction of rows where source *suffix* returned the bulk fallback."""
    if f"reasons_3_points__{suffix}" not in df.columns:
        return float("nan")
    is_zero = df.get(f"subclass__{suffix}") == "0"
    is_fallback_text = df[f"reasons_3_points__{suffix}"].astype(str).str.contains(
        "Insufficient information", na=False, regex=False,
    )
    return float((is_zero & is_fallback_text).mean())


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    print(f"Joined N = {len(df):,} (org_uuid intersection of all three sources)")

    summary: dict = {
        "n_total": int(len(df)),
        "axes": list(AXES.keys()),
        "pairs": [list(p) for p in PAIRS],
        "global": {},
        "fallback_rates_by_arm": {
            "baseline": _fallback_rate(df, "baseline"),
            "arm_a": _fallback_rate(df, "arm_a"),
            "arm_b": _fallback_rate(df, "arm_b"),
        },
    }

    for axis in AXES:
        summary["global"][axis] = {}
        for s1, s2 in PAIRS:
            col1, col2 = f"{axis}__{s1}", f"{axis}__{s2}"
            if col1 not in df.columns or col2 not in df.columns:
                continue
            metrics = _paired_metrics(df[col1], df[col2], axis)
            summary["global"][axis][f"{s1}__vs__{s2}"] = metrics
            _confusion_csv(df[col1], df[col2], axis, f"{s1}_vs_{s2}")

    strata_rows: list[dict] = []
    fallback_rows: list[dict] = []

    strata_specs = [
        ("fame_quartile", "fame_quartile"),
        ("baseline_subclass", "subclass__baseline"),
        ("baseline_cohort", "cohort__baseline"),
        ("baseline_conf_classification", "conf_classification__baseline"),
    ]

    for stratum_name, stratum_col in strata_specs:
        if stratum_col not in df.columns:
            continue
        groups = df.groupby(stratum_col, dropna=False)
        for stratum_value, sub in groups:
            for arm in ("baseline", "arm_a", "arm_b"):
                fallback_rows.append({
                    "stratum": stratum_name,
                    "stratum_value": str(stratum_value),
                    "arm": arm,
                    "n": int(len(sub)),
                    "fallback_rate": round(_fallback_rate(sub, arm), 4),
                })
            for axis in AXES:
                for s1, s2 in PAIRS:
                    col1, col2 = f"{axis}__{s1}", f"{axis}__{s2}"
                    if col1 not in sub.columns or col2 not in sub.columns:
                        continue
                    m = _paired_metrics(sub[col1], sub[col2], axis)
                    if m.get("n", 0) == 0:
                        continue
                    strata_rows.append({
                        "stratum": stratum_name,
                        "stratum_value": str(stratum_value),
                        "axis": axis,
                        "pair": f"{s1}__vs__{s2}",
                        **m,
                    })

    pd.DataFrame(strata_rows).to_csv(ANALYSIS_DIR / "stratified_metrics.csv", index=False)
    pd.DataFrame(fallback_rows).to_csv(ANALYSIS_DIR / "fallback_rates.csv", index=False)

    fame_rows = [r for r in strata_rows if r["stratum"] == "fame_quartile"]
    pd.DataFrame(fame_rows).to_csv(ANALYSIS_DIR / "kappa_by_fame_quartile.csv", index=False)

    summary["coverage"] = _build_coverage(int(len(df)))
    caveat = _data_caveat()
    if caveat is not None:
        summary["data_caveat"] = caveat
    summary["headlines"] = _build_headlines(summary["global"], strata_rows)

    out = ANALYSIS_DIR / "directness_metrics.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out}")
    print(f"Wrote {ANALYSIS_DIR / 'stratified_metrics.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'fallback_rates.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'kappa_by_fame_quartile.csv'}")
    print()
    cov = summary["coverage"]
    print("Coverage (row counts):")
    for key in ("n_master", "n_baseline", "n_arm_a", "n_arm_b", "n_joined"):
        val = cov.get(key)
        label = key.replace("n_", "")
        if val is None:
            print(f"  {label:12s}  n/a")
        else:
            print(f"  {label:12s}  {val:,}")
    if caveat is not None:
        print(
            f"  long_description nonempty: {caveat['long_description_nonempty_pct']:.2f}%"
        )
    print()
    hl = summary["headlines"]
    ba = hl["ai_native_baseline_vs_arm_a_global"]
    print("HEADLINE ai_native baseline vs arm_a (global):")
    print(f"  kappa={ba.get('kappa')}  agreement={ba.get('agreement')}  n={ba.get('n')}")
    ab = hl["ai_native_arm_a_vs_arm_b_fame"]
    print("HEADLINE ai_native arm_a vs arm_b by fame quartile:")
    for q in ("Q1", "Q4"):
        m = ab[q]
        print(f"  {q}  kappa={m.get('kappa')}  agreement={m.get('agreement')}  n={m.get('n')}")
    print(f"  kappa delta (Q4 - Q1) = {ab.get('kappa_delta_q4_minus_q1')}")
    print()
    print("Fallback rate (subclass=0 AND 'Insufficient information' in reasons):")
    for arm, rate in summary["fallback_rates_by_arm"].items():
        print(f"  {arm:12s}  {rate:.3%}" if not np.isnan(rate) else f"  {arm:12s}  n/a")


if __name__ == "__main__":
    main()
