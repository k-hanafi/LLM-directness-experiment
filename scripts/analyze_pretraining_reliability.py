"""Validate Baseline and Arm A classifications against evidence-grounded labels.

This script compares both the Baseline (Crunchbase text) and Arm A (pretraining
knowledge only) classifications against an external ground truth: the tavily-
grounded classifications produced by crawling company homepages and feeding that
evidence into the same classifier.

The analysis is restricted to the "evidence-only subset": companies where the
Tavily crawler returned non-empty website content (~22k).  The remaining ~22k
companies in the tavily production run were classified from Crunchbase text alone
(functionally identical to Baseline) and are excluded to avoid circularity.

Inputs:
    outputs/baseline/classified_baseline.csv
    outputs/arm_a/classified_arm_a.csv
    outputs/ground_truth/tavily_classifications.csv
    outputs/ground_truth/evidence_index.csv          (org_uuids with crawl data)
    outputs/analysis/fame_quartiles.csv              (optional, for stratification)

Outputs in outputs/analysis/:
    ground_truth_validation_metrics.json
    ground_truth_agreement_by_axis.csv
    ground_truth_confidence_comparison.csv
    ground_truth_agreement_by_fame.csv
    ground_truth_agreement_by_conf_tier.csv
    ground_truth_base_rate_decomposition.csv
    confusion_baseline_vs_tavily__<axis>.csv
    confusion_arm_a_vs_tavily__<axis>.csv

Statistical methodology:
    * Raw agreement rate (paired %)
    * Cohen's kappa (sklearn.metrics.cohen_kappa_score)
    * McNemar's test for binary axes
    * Stuart-Maxwell test for multi-class axes
    * Wilcoxon signed-rank test for paired confidence shifts
    * Per-tier agreement curves (discrimination within each arm)
    * Base-rate decomposition (class-conditional agreement)

Usage:
    python scripts/analyze_pretraining_reliability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.contingency_tables import SquareTable, mcnemar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

BASELINE_PATH = PROJECT_ROOT / "outputs" / "baseline" / "classified_baseline.csv"
ARM_A_PATH = PROJECT_ROOT / "outputs" / "arm_a" / "classified_arm_a.csv"
TAVILY_PATH = PROJECT_ROOT / "outputs" / "ground_truth" / "tavily_classifications.csv"
EVIDENCE_INDEX_PATH = PROJECT_ROOT / "outputs" / "ground_truth" / "evidence_index.csv"
FAME_PATH = ANALYSIS_DIR / "fame_quartiles.csv"

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
    ("baseline", "tavily"),
    ("arm_a", "tavily"),
]


def _load(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df = df.rename(columns={"CompanyID": "org_uuid"})
    keep = ["org_uuid"] + list(AXES.keys()) + ["conf_classification"]
    df = df[[c for c in keep if c in df.columns]]
    rename = {c: f"{c}__{suffix}" for c in df.columns if c != "org_uuid"}
    return df.rename(columns=rename)


def load_joined() -> pd.DataFrame:
    """Inner-join Baseline, Arm A, and Tavily GT on the evidence-only subset."""
    if not EVIDENCE_INDEX_PATH.exists():
        print(f"ERROR: missing {EVIDENCE_INDEX_PATH}", file=sys.stderr)
        sys.exit(1)

    evidence_ids = set(pd.read_csv(EVIDENCE_INDEX_PATH)["org_uuid"].astype(str))

    base = _load(BASELINE_PATH, "baseline")
    arm_a = _load(ARM_A_PATH, "arm_a")
    tavily = _load(TAVILY_PATH, "tavily")

    df = base.merge(arm_a, on="org_uuid").merge(tavily, on="org_uuid")
    df = df[df["org_uuid"].isin(evidence_ids)].reset_index(drop=True)

    if FAME_PATH.exists():
        fame = pd.read_csv(FAME_PATH)[["org_uuid", "fame_quartile"]]
        fame["org_uuid"] = fame["org_uuid"].astype(str)
        df = df.merge(fame, on="org_uuid", how="left")
    else:
        print(f"WARNING: {FAME_PATH} missing; fame strata skipped", file=sys.stderr)
        df["fame_quartile"] = "ALL"

    return df


def _paired_metrics(y1: pd.Series, y2: pd.Series, axis: str) -> dict:
    """Agreement rate, kappa, and directional test for one (axis, pair)."""
    mask = y1.notna() & y2.notna()
    y1, y2 = y1[mask], y2[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}

    agree = float((y1 == y2).mean())

    info = AXES[axis]
    labels = info["labels"]
    try:
        kappa = float(cohen_kappa_score(y1, y2, labels=labels))
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
    out_df = pd.DataFrame(cm, index=labels, columns=labels)
    out_df.index.name = "row=arm"
    out_df.columns.name = "col=tavily_gt"
    out = ANALYSIS_DIR / f"confusion_{name}__{axis}.csv"
    out_df.to_csv(out)
    return out


def _confidence_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Paired confidence stats for all three sources."""
    rows = []
    sources = [
        ("baseline", "conf_classification__baseline"),
        ("arm_a", "conf_classification__arm_a"),
        ("tavily", "conf_classification__tavily"),
    ]

    for name, col in sources:
        vals = df[col].dropna().astype(int)
        rows.append({
            "source": name,
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 3),
            "median": float(vals.median()),
            "std": round(float(vals.std()), 3),
            "pct_conf_1": round(float((vals == 1).mean()), 4),
            "pct_conf_2": round(float((vals == 2).mean()), 4),
            "pct_conf_3": round(float((vals == 3).mean()), 4),
            "pct_conf_4": round(float((vals == 4).mean()), 4),
            "pct_conf_5": round(float((vals == 5).mean()), 4),
        })

    # Wilcoxon signed-rank: each arm vs tavily
    for arm_name, arm_col in [("baseline", "conf_classification__baseline"),
                               ("arm_a", "conf_classification__arm_a")]:
        mask = df[arm_col].notna() & df["conf_classification__tavily"].notna()
        arm_vals = df.loc[mask, arm_col].astype(int)
        tav_vals = df.loc[mask, "conf_classification__tavily"].astype(int)
        diff = tav_vals - arm_vals
        nonzero = diff[diff != 0]
        if len(nonzero) > 10:
            stat, pval = wilcoxon(nonzero)
            rows.append({
                "source": f"wilcoxon_{arm_name}_vs_tavily",
                "n": int(len(nonzero)),
                "mean": round(float(diff.mean()), 3),
                "median": float(diff.median()),
                "std": round(float(diff.std()), 3),
                "pct_conf_1": None,
                "pct_conf_2": None,
                "pct_conf_3": None,
                "pct_conf_4": None,
                "pct_conf_5": None,
                "wilcoxon_stat": round(float(stat), 2),
                "wilcoxon_p": float(f"{pval:.4g}"),
            })

    return pd.DataFrame(rows)


def _agreement_by_conf_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tier agreement curve for both arms against tavily GT."""
    rows = []
    for arm in ("baseline", "arm_a"):
        conf_col = f"conf_classification__{arm}"
        for conf_val in range(1, 6):
            sub = df[df[conf_col] == conf_val]
            if len(sub) == 0:
                continue
            n = len(sub)
            for axis in AXES:
                arm_col = f"{axis}__{arm}"
                gt_col = f"{axis}__tavily"
                if arm_col not in sub.columns or gt_col not in sub.columns:
                    continue
                mask = sub[arm_col].notna() & sub[gt_col].notna()
                if mask.sum() == 0:
                    continue
                agree = float((sub.loc[mask, arm_col] == sub.loc[mask, gt_col]).mean())
                rows.append({
                    "arm": arm,
                    "conf_tier": conf_val,
                    "axis": axis,
                    "n": int(mask.sum()),
                    "agreement": round(agree, 4),
                })
    return pd.DataFrame(rows)


def _base_rate_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Class-conditional agreement: separates trivial TN agreement from TP."""
    rows = []
    for arm in ("baseline", "arm_a"):
        arm_col = f"ai_native__{arm}"
        gt_col = "ai_native__tavily"
        mask = df[arm_col].notna() & df[gt_col].notna()
        sub = df[mask]

        gt_positive = sub[sub[gt_col] == 1]
        gt_negative = sub[sub[gt_col] == 0]

        # Agreement among GT-positive (AI-native companies)
        if len(gt_positive) > 0:
            tp_agree = float((gt_positive[arm_col] == 1).mean())
            rows.append({
                "arm": arm,
                "gt_class": "ai_native=1",
                "n": int(len(gt_positive)),
                "agreement": round(tp_agree, 4),
                "interpretation": "recall (sensitivity)",
            })

        # Agreement among GT-negative (not AI-native)
        if len(gt_negative) > 0:
            tn_agree = float((gt_negative[arm_col] == 0).mean())
            rows.append({
                "arm": arm,
                "gt_class": "ai_native=0",
                "n": int(len(gt_negative)),
                "agreement": round(tn_agree, 4),
                "interpretation": "specificity",
            })

        # Among arm-positive predictions: precision
        arm_positive = sub[sub[arm_col] == 1]
        if len(arm_positive) > 0:
            precision = float((arm_positive[gt_col] == 1).mean())
            rows.append({
                "arm": arm,
                "gt_class": "arm_predicts_1",
                "n": int(len(arm_positive)),
                "agreement": round(precision, 4),
                "interpretation": "precision (PPV)",
            })

    return pd.DataFrame(rows)


def _fame_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement metrics stratified by fame quartile for both pairs."""
    if "fame_quartile" not in df.columns or df["fame_quartile"].isna().all():
        return pd.DataFrame()

    rows = []
    for quartile, sub in df.groupby("fame_quartile", dropna=False):
        for arm in ("baseline", "arm_a"):
            for axis in AXES:
                arm_col = f"{axis}__{arm}"
                gt_col = f"{axis}__tavily"
                if arm_col not in sub.columns or gt_col not in sub.columns:
                    continue
                m = _paired_metrics(sub[arm_col], sub[gt_col], axis)
                if m.get("n", 0) == 0:
                    continue
                rows.append({
                    "fame_quartile": str(quartile),
                    "arm": arm,
                    "axis": axis,
                    **m,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    print(f"Evidence-only joined N = {len(df):,}")

    # --- Global metrics for both pairs ---
    axis_rows = []
    summary: dict = {
        "n_total": int(len(df)),
        "pairs": [list(p) for p in PAIRS],
        "global": {},
    }

    for axis in AXES:
        summary["global"][axis] = {}
        for s1, _ in PAIRS:
            col1 = f"{axis}__{s1}"
            col2 = f"{axis}__tavily"
            if col1 not in df.columns or col2 not in df.columns:
                continue
            metrics = _paired_metrics(df[col1], df[col2], axis)
            pair_label = f"{s1}__vs__tavily"
            summary["global"][axis][pair_label] = metrics
            _confusion_csv(df[col1], df[col2], axis, f"{s1}_vs_tavily")
            axis_rows.append({"pair": pair_label, "axis": axis, **metrics})

    pd.DataFrame(axis_rows).to_csv(
        ANALYSIS_DIR / "ground_truth_agreement_by_axis.csv", index=False
    )

    # --- Accuracy gap (headline) ---
    base_agree = summary["global"].get("ai_native", {}).get(
        "baseline__vs__tavily", {}
    ).get("agreement")
    arm_a_agree = summary["global"].get("ai_native", {}).get(
        "arm_a__vs__tavily", {}
    ).get("agreement")
    if base_agree is not None and arm_a_agree is not None:
        summary["accuracy_gap_ai_native"] = round(base_agree - arm_a_agree, 4)
    else:
        summary["accuracy_gap_ai_native"] = None

    # --- Confidence comparison ---
    conf_df = _confidence_comparison(df)
    conf_df.to_csv(
        ANALYSIS_DIR / "ground_truth_confidence_comparison.csv", index=False
    )
    summary["confidence"] = {
        "baseline_mean": conf_df.loc[conf_df["source"] == "baseline", "mean"].iloc[0],
        "arm_a_mean": conf_df.loc[conf_df["source"] == "arm_a", "mean"].iloc[0],
        "tavily_mean": conf_df.loc[conf_df["source"] == "tavily", "mean"].iloc[0],
    }

    # --- Per-tier agreement (discrimination) ---
    tier_df = _agreement_by_conf_tier(df)
    tier_df.to_csv(
        ANALYSIS_DIR / "ground_truth_agreement_by_conf_tier.csv", index=False
    )

    # --- Base-rate decomposition ---
    baserate_df = _base_rate_decomposition(df)
    baserate_df.to_csv(
        ANALYSIS_DIR / "ground_truth_base_rate_decomposition.csv", index=False
    )

    # --- Fame stratification ---
    fame_df = _fame_stratified(df)
    if not fame_df.empty:
        fame_df.to_csv(
            ANALYSIS_DIR / "ground_truth_agreement_by_fame.csv", index=False
        )

    # --- Headlines ---
    summary["headlines"] = {
        "baseline_vs_tavily_ai_native": summary["global"].get("ai_native", {}).get(
            "baseline__vs__tavily"
        ),
        "arm_a_vs_tavily_ai_native": summary["global"].get("ai_native", {}).get(
            "arm_a__vs__tavily"
        ),
        "accuracy_gap_ai_native_pp": (
            f"{summary['accuracy_gap_ai_native'] * 100:.1f}pp"
            if summary["accuracy_gap_ai_native"] is not None
            else None
        ),
    }

    out = ANALYSIS_DIR / "ground_truth_validation_metrics.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    # --- Print summary ---
    print()
    print("=" * 60)
    print("GROUND-TRUTH VALIDATION RESULTS")
    print("=" * 60)
    print()
    print(f"Evidence-only sample: n = {len(df):,}")
    print()
    print("--- Binary ai_native agreement with ground truth ---")
    for pair_label in ["baseline__vs__tavily", "arm_a__vs__tavily"]:
        m = summary["global"].get("ai_native", {}).get(pair_label, {})
        print(
            f"  {pair_label:30s}  agreement={m.get('agreement')}  "
            f"kappa={m.get('kappa')}  p={m.get('p_value')}"
        )
    print(f"  {'accuracy_gap':30s}  {summary['accuracy_gap_ai_native']}")
    print()
    print("--- Confidence (information rent) ---")
    print(f"  Arm A mean:     {summary['confidence']['arm_a_mean']}")
    print(f"  Baseline mean:  {summary['confidence']['baseline_mean']}")
    print(f"  Tavily mean:    {summary['confidence']['tavily_mean']}")
    print()
    print("--- Base-rate decomposition (ai_native) ---")
    for _, row in baserate_df.iterrows():
        print(
            f"  {row['arm']:10s} | {row['gt_class']:20s} | "
            f"agreement={row['agreement']:.4f} (n={row['n']}) "
            f"[{row['interpretation']}]"
        )
    print()
    if not fame_df.empty:
        print("--- Fame-stratified kappa (ai_native, baseline vs tavily) ---")
        sub = fame_df[
            (fame_df["arm"] == "baseline") & (fame_df["axis"] == "ai_native")
        ]
        for _, row in sub.iterrows():
            print(
                f"  {row['fame_quartile']:4s}  kappa={row.get('kappa')}  "
                f"agreement={row.get('agreement')}  n={row.get('n')}"
            )
        print()
        print("--- Fame-stratified kappa (ai_native, arm_a vs tavily) ---")
        sub = fame_df[
            (fame_df["arm"] == "arm_a") & (fame_df["axis"] == "ai_native")
        ]
        for _, row in sub.iterrows():
            print(
                f"  {row['fame_quartile']:4s}  kappa={row.get('kappa')}  "
                f"agreement={row.get('agreement')}  n={row.get('n')}"
            )

    print()
    print(f"Wrote {out}")
    print(f"Wrote {ANALYSIS_DIR / 'ground_truth_agreement_by_axis.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'ground_truth_confidence_comparison.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'ground_truth_agreement_by_conf_tier.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'ground_truth_base_rate_decomposition.csv'}")
    if not fame_df.empty:
        print(f"Wrote {ANALYSIS_DIR / 'ground_truth_agreement_by_fame.csv'}")


if __name__ == "__main__":
    main()
