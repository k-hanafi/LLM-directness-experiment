"""Render a self-contained HTML dashboard diagnosing pretraining bias and data leakage.

Loads the three classified CSVs and fame quartiles directly, computes all
metrics in-memory, and writes a Plotly HTML file with four sections:

  1. Agreement Rates   -- how much do Arms A/B agree with Baseline?
  2. Fame as Leakage   -- does agreement increase with fame?
  3. Confidence         -- does the model know what it doesn't know?
  4. Fallback Behavior  -- when does the model give up?

FILTERING: Rows where an arm has conf_classification < 2 are excluded from
agreement analysis for that arm. These low-confidence rows are pure guesses
that inflate agreement via base-rate coincidence (both arms default to
ai_native=0). Section 4 uses the full dataset since it measures fallback.

Inputs:
    outputs/baseline/classified_baseline.csv
    outputs/arm_a/classified_arm_a.csv
    outputs/arm_b/classified_arm_b.csv
    outputs/analysis/fame_quartiles.csv

Writes:
    outputs/analysis/directness_dashboard.html

Usage:
    python scripts/build_directness_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

BASELINE_PATH = PROJECT_ROOT / "outputs" / "baseline" / "classified_baseline.csv"
ARM_A_PATH = PROJECT_ROOT / "outputs" / "arm_a" / "classified_arm_a.csv"
ARM_B_PATH = PROJECT_ROOT / "outputs" / "arm_b" / "classified_arm_b.csv"
FAME_PATH = ANALYSIS_DIR / "fame_quartiles.csv"
DATAVIZ_DIR = PROJECT_ROOT / "data visualization"
DASH_OUT = DATAVIZ_DIR / "directness_dashboard.html"

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

PAIR_LABELS = {
    "baseline__vs__arm_a": "Baseline vs Arm A",
    "baseline__vs__arm_b": "Baseline vs Arm B",
    "arm_a__vs__arm_b": "Arm A vs Arm B",
}

PAIR_COLORS = {
    "Baseline vs Arm A": "#2563eb",
    "Baseline vs Arm B": "#dc2626",
    "Arm A vs Arm B": "#16a34a",
}

ARM_COLORS = {
    "Baseline": "#2563eb",
    "Arm A": "#dc2626",
    "Arm B": "#16a34a",
}

QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]

MIN_CONF = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_arm(path: Path, suffix: str) -> pd.DataFrame:
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
    base = _load_arm(BASELINE_PATH, "baseline")
    arm_a = _load_arm(ARM_A_PATH, "arm_a")
    arm_b = _load_arm(ARM_B_PATH, "arm_b")
    df = base.merge(arm_a, on="org_uuid").merge(arm_b, on="org_uuid")
    if FAME_PATH.exists():
        fame = pd.read_csv(FAME_PATH)[["org_uuid", "fame_quartile"]]
        df = df.merge(fame, on="org_uuid", how="left")
    else:
        print(f"WARNING: {FAME_PATH} missing; fame charts will be empty.", file=sys.stderr)
        df["fame_quartile"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Fallback detection and pair filtering
# ---------------------------------------------------------------------------

def _is_fallback(df: pd.DataFrame, suffix: str) -> pd.Series:
    """True when the arm's confidence is below MIN_CONF (pure guessing)."""
    conf_col = f"conf_classification__{suffix}"
    return df[conf_col] < MIN_CONF


def _filter_for_pair(df: pd.DataFrame, s1: str, s2: str) -> pd.DataFrame:
    """Exclude rows where either non-baseline arm is fallback."""
    mask = pd.Series(True, index=df.index)
    for s in (s1, s2):
        if s != "baseline":
            mask &= ~_is_fallback(df, s)
    return df[mask]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _agreement(y1: pd.Series, y2: pd.Series) -> float:
    mask = y1.notna() & y2.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((y1[mask] == y2[mask]).mean())


def _kappa(y1: pd.Series, y2: pd.Series, labels: list | None = None) -> float:
    mask = y1.notna() & y2.notna()
    if mask.sum() == 0:
        return float("nan")
    try:
        if labels:
            return float(cohen_kappa_score(y1[mask], y2[mask], labels=labels))
        return float(cohen_kappa_score(y1[mask], y2[mask]))
    except Exception:
        return float("nan")


def _pair_key(s1: str, s2: str) -> str:
    return f"{s1}__vs__{s2}"


def _pair_label(key: str) -> str:
    return PAIR_LABELS.get(key, key.replace("__vs__", " vs "))


# ---------------------------------------------------------------------------
# Section 1: Agreement Rates (filtered)
# ---------------------------------------------------------------------------

def _section1_agreement(df: pd.DataFrame) -> tuple[str, dict]:
    rows = []
    for axis, info in AXES.items():
        for s1, s2 in PAIRS:
            col1, col2 = f"{axis}__{s1}", f"{axis}__{s2}"
            if col1 not in df.columns or col2 not in df.columns:
                continue
            filtered = _filter_for_pair(df, s1, s2)
            agree = _agreement(filtered[col1], filtered[col2])
            kap = _kappa(filtered[col1], filtered[col2], info["labels"])
            pk = _pair_key(s1, s2)
            rows.append({
                "Axis": axis,
                "Pair": _pair_label(pk),
                "pair_key": pk,
                "Agreement": round(agree, 4),
                "Kappa": round(kap, 4) if not np.isnan(kap) else None,
                "n": len(filtered),
            })

    tbl = pd.DataFrame(rows)

    fig_agree = px.bar(
        tbl, x="Axis", y="Agreement", color="Pair", barmode="group",
        title="Agreement Rate by Classification Axis (fallback rows excluded)",
        labels={"Agreement": "Agreement Rate"},
        hover_data=["Kappa", "n"],
        color_discrete_map=PAIR_COLORS,
    )
    fig_agree.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_agree.update_layout(height=400, legend_title_text="Pair")

    fig_kappa = px.bar(
        tbl, x="Axis", y="Kappa", color="Pair", barmode="group",
        title="Cohen's Kappa by Classification Axis (fallback rows excluded)",
        labels={"Kappa": "Cohen's kappa"},
        hover_data=["Agreement", "n"],
        color_discrete_map=PAIR_COLORS,
    )
    fig_kappa.update_yaxes(range=[-0.1, 1.05])
    fig_kappa.update_layout(height=400, legend_title_text="Pair")

    html = (
        fig_agree.to_html(full_html=False, include_plotlyjs=False, div_id="s1-agree")
        + fig_kappa.to_html(full_html=False, include_plotlyjs=False, div_id="s1-kappa")
    )

    headlines = {}
    for pk_key in ("baseline__vs__arm_a", "baseline__vs__arm_b"):
        row = tbl[(tbl["Axis"] == "ai_native") & (tbl["pair_key"] == pk_key)]
        if not row.empty:
            r = row.iloc[0]
            headlines[pk_key] = {
                "agreement": r["Agreement"],
                "kappa": r["Kappa"],
                "n": int(r["n"]),
            }

    return html, headlines


# ---------------------------------------------------------------------------
# Section 2: Fame as a Leakage Indicator (filtered)
# ---------------------------------------------------------------------------

def _section2_fame(df: pd.DataFrame) -> str:
    has_fame = df["fame_quartile"].notna().any()
    if not has_fame:
        return "<p><em>Fame quartiles not available.</em></p>"

    fame_df = df[df["fame_quartile"].notna()].copy()
    fame_df["fame_quartile"] = pd.Categorical(
        fame_df["fame_quartile"], categories=QUARTILE_ORDER, ordered=True,
    )

    # 2a & 2b: agreement rate and kappa by fame quartile (filtered)
    rows = []
    for q in QUARTILE_ORDER:
        q_df = fame_df[fame_df["fame_quartile"] == q]
        if q_df.empty:
            continue
        for s1, s2 in PAIRS:
            pk = _pair_key(s1, s2)
            filtered = _filter_for_pair(q_df, s1, s2)
            if filtered.empty:
                continue
            col1, col2 = f"ai_native__{s1}", f"ai_native__{s2}"
            agree = _agreement(filtered[col1], filtered[col2])
            kap = _kappa(filtered[col1], filtered[col2])
            rows.append({
                "Fame Quartile": q,
                "Pair": _pair_label(pk),
                "Agreement": round(agree, 4),
                "Kappa": round(kap, 4) if not np.isnan(kap) else None,
                "n": int(len(filtered)),
            })

    fame_tbl = pd.DataFrame(rows)

    fig_agree = px.line(
        fame_tbl, x="Fame Quartile", y="Agreement", color="Pair",
        markers=True, hover_data=["n", "Kappa"],
        title="ai_native Agreement by Fame Quartile (fallback excluded)",
        labels={"Agreement": "Agreement Rate",
                "Fame Quartile": "Fame Quartile (Q1=obscure, Q4=famous)"},
        color_discrete_map=PAIR_COLORS,
    )
    fig_agree.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_agree.update_layout(height=420, legend_title_text="Pair")

    fig_kappa = px.line(
        fame_tbl, x="Fame Quartile", y="Kappa", color="Pair",
        markers=True, hover_data=["n", "Agreement"],
        title="ai_native Cohen's Kappa by Fame Quartile (fallback excluded)",
        labels={"Kappa": "Cohen's kappa",
                "Fame Quartile": "Fame Quartile (Q1=obscure, Q4=famous)"},
        color_discrete_map=PAIR_COLORS,
    )
    fig_kappa.update_yaxes(range=[-0.1, 1.05])
    fig_kappa.update_layout(height=420, legend_title_text="Pair")

    # 2c: mean confidence by fame quartile, only on non-fallback rows per arm
    conf_rows = []
    for q in QUARTILE_ORDER:
        q_df = fame_df[fame_df["fame_quartile"] == q]
        if q_df.empty:
            continue
        for arm in ("baseline", "arm_a", "arm_b"):
            if arm == "baseline":
                sub = q_df
            else:
                sub = q_df[~_is_fallback(q_df, arm)]
            conf_col = f"conf_classification__{arm}"
            if conf_col not in sub.columns or sub.empty:
                continue
            conf_rows.append({
                "Fame Quartile": q,
                "Arm": arm.replace("_", " ").title(),
                "Mean Confidence": round(float(sub[conf_col].mean()), 3),
                "n": int(len(sub)),
            })

    conf_tbl = pd.DataFrame(conf_rows)
    fig_conf = px.line(
        conf_tbl, x="Fame Quartile", y="Mean Confidence", color="Arm",
        markers=True, hover_data=["n"],
        title="Mean Confidence by Fame Quartile (fallback excluded for Arms)",
        labels={"Mean Confidence": "Mean conf_classification (1-5)",
                "Fame Quartile": "Fame Quartile (Q1=obscure, Q4=famous)"},
        color_discrete_map=ARM_COLORS,
    )
    fig_conf.update_yaxes(range=[0.8, 5.2])
    fig_conf.update_layout(height=420, legend_title_text="Arm")

    return (
        fig_agree.to_html(full_html=False, include_plotlyjs=False, div_id="s2-agree")
        + fig_kappa.to_html(full_html=False, include_plotlyjs=False, div_id="s2-kappa")
        + fig_conf.to_html(full_html=False, include_plotlyjs=False, div_id="s2-conf")
    )


# ---------------------------------------------------------------------------
# Section 3: Confidence Analysis (filtered)
# ---------------------------------------------------------------------------

def _section3_confidence(df: pd.DataFrame) -> str:
    # 3a: full confidence distribution (unfiltered) to show the raw picture
    conf_rows = []
    for arm in ("baseline", "arm_a", "arm_b"):
        conf_col = f"conf_classification__{arm}"
        if conf_col not in df.columns:
            continue
        counts = df[conf_col].value_counts(normalize=True).sort_index()
        for level, pct in counts.items():
            conf_rows.append({
                "Arm": arm.replace("_", " ").title(),
                "Confidence": int(level),
                "Proportion": round(float(pct), 4),
            })

    conf_tbl = pd.DataFrame(conf_rows)
    conf_tbl["Confidence"] = conf_tbl["Confidence"].astype(str)

    fig_dist = px.bar(
        conf_tbl, x="Arm", y="Proportion", color="Confidence",
        barmode="stack",
        title="Confidence Distribution by Arm (all rows, unfiltered)",
        labels={"Proportion": "Proportion of rows"},
        category_orders={"Confidence": ["1", "2", "3", "4", "5"]},
        color_discrete_sequence=["#dee2e6", "#a5d8ff", "#4dabf7", "#1c7ed6", "#1864ab"],
    )
    fig_dist.update_yaxes(tickformat=".0%")
    fig_dist.update_layout(height=400, legend_title_text="conf_classification")

    # 3b: agreement rate by confidence level for Arm A
    # shows whether higher confidence = more agreement with baseline
    conf_agree_rows = []
    for arm, arm_label in [("arm_a", "Arm A"), ("arm_b", "Arm B")]:
        ai_col = f"ai_native__{arm}"
        conf_col = f"conf_classification__{arm}"
        base_col = "ai_native__baseline"
        if any(c not in df.columns for c in (ai_col, conf_col, base_col)):
            continue
        for conf_level in range(1, 6):
            sub = df[df[conf_col] == conf_level]
            if sub.empty:
                continue
            agree = (sub[ai_col] == sub[base_col]).mean()
            conf_agree_rows.append({
                "Arm": arm_label,
                "Confidence Level": str(conf_level),
                "Agreement with Baseline": round(float(agree), 4),
                "n": int(len(sub)),
            })

    conf_agree_tbl = pd.DataFrame(conf_agree_rows)
    fig_conf_agree = px.bar(
        conf_agree_tbl, x="Confidence Level", y="Agreement with Baseline",
        color="Arm", barmode="group", hover_data=["n"],
        title="Agreement Rate with Baseline by Confidence Level",
        labels={"Agreement with Baseline": "Agreement Rate",
                "Confidence Level": "conf_classification"},
        color_discrete_map={"Arm A": "#dc2626", "Arm B": "#16a34a"},
    )
    fig_conf_agree.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_conf_agree.update_layout(height=400, legend_title_text="Arm")

    return (
        fig_dist.to_html(full_html=False, include_plotlyjs=False, div_id="s3-dist")
        + fig_conf_agree.to_html(full_html=False, include_plotlyjs=False, div_id="s3-cal")
    )


# ---------------------------------------------------------------------------
# Section 4: Fallback Rates (full data -- this section measures fallback)
# ---------------------------------------------------------------------------

def _section4_fallback(df: pd.DataFrame) -> str:
    has_fame = df["fame_quartile"].notna().any()

    fb_rows = []
    groups = QUARTILE_ORDER if has_fame else ["ALL"]
    for q in groups:
        if q == "ALL":
            sub = df
        else:
            sub = df[df["fame_quartile"] == q]
        if sub.empty:
            continue
        for arm in ("baseline", "arm_a", "arm_b"):
            fb = _is_fallback(sub, arm)
            fb_rows.append({
                "Fame Quartile": q,
                "Arm": arm.replace("_", " ").title(),
                "Fallback Rate": round(float(fb.mean()), 4),
                "n": int(len(sub)),
            })

    fb_tbl = pd.DataFrame(fb_rows)
    fig_fb = px.bar(
        fb_tbl, x="Fame Quartile", y="Fallback Rate", color="Arm",
        barmode="group", hover_data=["n"],
        title="Fallback Rate (conf < 2) by Fame Quartile",
        labels={"Fallback Rate": "Fraction classified as fallback",
                "Fame Quartile": "Fame Quartile (Q1=obscure, Q4=famous)"},
        color_discrete_map=ARM_COLORS,
    )
    fig_fb.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_fb.update_layout(height=420, legend_title_text="Arm")

    # For rows where arm fell back but baseline did NOT, show baseline subclass
    lost_rows = []
    for arm, arm_label in [("arm_a", "Arm A"), ("arm_b", "Arm B")]:
        arm_fb = _is_fallback(df, arm)
        base_no_fb = ~_is_fallback(df, "baseline")
        lost = df[arm_fb & base_no_fb]
        if lost.empty:
            continue
        counts = lost["subclass__baseline"].value_counts()
        for sub, cnt in counts.items():
            lost_rows.append({
                "Arm That Fell Back": arm_label,
                "Baseline Subclass": str(sub),
                "Count": int(cnt),
            })

    if lost_rows:
        lost_tbl = pd.DataFrame(lost_rows)
        fig_lost = px.bar(
            lost_tbl, x="Count", y="Baseline Subclass", color="Arm That Fell Back",
            barmode="group", orientation="h",
            title="Lost Classifications: What Baseline Assigned When Arms Fell Back",
            labels={"Count": "Number of companies",
                    "Baseline Subclass": "Baseline subclass"},
        )
        fig_lost.update_layout(
            height=max(350, 30 * lost_tbl["Baseline Subclass"].nunique() + 150),
            legend_title_text="Arm",
        )
        lost_html = fig_lost.to_html(full_html=False, include_plotlyjs=False, div_id="s4-lost")
    else:
        lost_html = "<p><em>No fallback discrepancies found.</em></p>"

    return (
        fig_fb.to_html(full_html=False, include_plotlyjs=False, div_id="s4-fb")
        + lost_html
    )


# ---------------------------------------------------------------------------
# Headline callout bar
# ---------------------------------------------------------------------------

def _headline_bar(df: pd.DataFrame, headlines: dict) -> str:
    n_total = len(df)
    ba = headlines.get("baseline__vs__arm_a", {})
    bb = headlines.get("baseline__vs__arm_b", {})

    ba_agree = ba.get("agreement")
    ba_kappa = ba.get("kappa")
    ba_n = ba.get("n", 0)
    bb_agree = bb.get("agreement")
    bb_kappa = bb.get("kappa")
    bb_n = bb.get("n", 0)

    fb_a = float(_is_fallback(df, "arm_a").mean())
    fb_b = float(_is_fallback(df, "arm_b").mean())

    def _fmt_pct(v: float | None) -> str:
        return f"{v:.1%}" if v is not None else "n/a"

    def _fmt_k(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "n/a"

    return f"""<div class="metrics-bar">
  <div class="metric-card">
    <div class="metric-value">{n_total:,}</div>
    <div class="metric-label">Total companies</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{_fmt_pct(fb_a)} / {_fmt_pct(fb_b)}</div>
    <div class="metric-label">Fallback rate<br>Arm A / Arm B</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{ba_n:,}</div>
    <div class="metric-label">Arm A substantive<br>rows analyzed</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{_fmt_pct(ba_agree)}</div>
    <div class="metric-label">Baseline vs Arm A<br>ai_native agreement</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{_fmt_k(ba_kappa)}</div>
    <div class="metric-label">Baseline vs Arm A<br>Cohen's kappa</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{bb_n:,}</div>
    <div class="metric-label">Arm B substantive<br>rows analyzed</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{_fmt_pct(bb_agree)}</div>
    <div class="metric-label">Baseline vs Arm B<br>ai_native agreement</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{_fmt_k(bb_kappa)}</div>
    <div class="metric-label">Baseline vs Arm B<br>Cohen's kappa</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Assemble HTML
# ---------------------------------------------------------------------------

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
  max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.6;
}
h1 { border-bottom: 2px solid #1864ab; padding-bottom: 0.3em; color: #1864ab; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.2em; margin-top: 2.5em; }
.hero {
  background: #f8f9fa; padding: 1em 1.4em; border-left: 4px solid #4c6ef5;
  margin: 1em 0 2em 0; font-size: 0.95em;
}
.metrics-bar {
  display: flex; flex-wrap: wrap; gap: 0.8em; margin: 1.5em 0;
}
.metric-card {
  flex: 1; min-width: 130px; background: #f1f3f5; border-radius: 8px;
  padding: 0.8em; text-align: center;
}
.metric-value { font-size: 1.4em; font-weight: 700; color: #1864ab; }
.metric-label { font-size: 0.75em; color: #666; margin-top: 0.3em; line-height: 1.3; }
.section-note { color: #666; font-size: 0.9em; margin-bottom: 0.5em; }
.methodology { font-size: 0.9em; color: #555; }
.methodology li { margin-bottom: 0.4em; }
"""


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    DATAVIZ_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    n = len(df)

    fb_a_n = int(_is_fallback(df, "arm_a").sum())
    fb_b_n = int(_is_fallback(df, "arm_b").sum())
    print(f"Loaded joined data: {n:,} companies")
    print(f"Arm A fallback (conf < {MIN_CONF}): {fb_a_n:,} ({fb_a_n/n:.1%})")
    print(f"Arm B fallback (conf < {MIN_CONF}): {fb_b_n:,} ({fb_b_n/n:.1%})")
    print(f"Arm A substantive rows: {n - fb_a_n:,}")
    print(f"Arm B substantive rows: {n - fb_b_n:,}")

    s1_html, headlines = _section1_agreement(df)
    s2_html = _section2_fame(df)
    s3_html = _section3_confidence(df)
    s4_html = _section4_fallback(df)
    headline_bar = _headline_bar(df, headlines)

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Directness Experiment: Pretraining Bias and Data Leakage</title>
<style>{CSS}</style>
{plotly_cdn}
</head>
<body>

<h1>Directness Experiment: Diagnosing Pretraining Bias and Data Leakage</h1>

<div class="hero">
<strong>Research question.</strong> When the classifier receives only a company name
(Arm A) or only an address with an anonymized name (Arm B) instead of full Crunchbase
fields (Baseline), does it still produce the same classification? High agreement
signals that the model is classifying from memorized pretraining knowledge rather than
the provided inputs. If agreement rises with company fame, that confirms data leakage:
famous companies are more likely to exist in the training data.
</div>

{headline_bar}

<h2>1. Agreement Rates</h2>
<p class="section-note">How much do Arms A and B agree with the Baseline when the arm
actually attempted a substantive classification? Only non-fallback rows are included.
Hover for sample sizes (n).</p>
{s1_html}

<h2>2. Fame as a Leakage Indicator</h2>
<p class="section-note">If agreement increases from Q1 (obscure) to Q4 (famous), the model
is drawing on memorized knowledge about well-known companies. An upward slope is
evidence of pretraining data leakage. Only non-fallback rows per pair are included.</p>
{s2_html}

<h2>3. Confidence Analysis</h2>
<p class="section-note">The confidence distribution (unfiltered) shows why filtering is necessary:
Arms A/B are dominated by conf=1. The agreement-by-confidence chart answers: does
higher confidence actually predict agreement with the Baseline?</p>
{s3_html}

<h2>4. Fallback Behavior</h2>
<p class="section-note">Uses the full dataset. Fallback = conf_classification &lt; {MIN_CONF}.
The "lost classifications" chart shows what the Baseline classified companies as when
the arm fell back to guessing.</p>
{s4_html}

<h2>Methodology</h2>
<ul class="methodology">
  <li><strong>Baseline</strong> uses <code>prompts/baseline_prompt.txt</code> (full
      Crunchbase inputs). <strong>Arm A</strong> uses <code>prompts/arm_a_prompt.txt</code>
      (real name + address only). <strong>Arm B</strong> uses
      <code>prompts/arm_b_prompt.txt</code> (anonymized name + address only).</li>
  <li>All three arms share the same model (<code>gpt-5.4-nano</code>), pipeline code,
      schema, and timing. Only the user-message fields differ.</li>
  <li><strong>Fallback filter:</strong> rows with <code>conf_classification &lt; {MIN_CONF}</code>
      are treated as pure guessing. These are excluded from agreement metrics (Sections
      1-3) to avoid base-rate inflation. The baseline is never filtered because it had
      full input data.</li>
  <li>Fame proxy is a composite z-score of Crunchbase rank (inverted), log funding,
      log funding rounds, and URL presence. Q1 = most obscure, Q4 = most famous.</li>
  <li>Cohen's kappa via <code>sklearn.metrics.cohen_kappa_score</code>.</li>
  <li>N = {n:,} total companies (inner join of all three arms).</li>
</ul>

</body>
</html>"""

    DASH_OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {DASH_OUT}")


if __name__ == "__main__":
    main()
