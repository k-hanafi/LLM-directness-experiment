"""Render a self-contained HTML dashboard summarising the directness audit.

Reads:
    outputs/analysis/directness_metrics.json
    outputs/analysis/stratified_metrics.csv
    outputs/analysis/fallback_rates.csv
    outputs/analysis/fame_quartiles.csv
    outputs/analysis/confusion_*.csv

Writes:
    outputs/analysis/directness_dashboard.html

Sections:
  1. Hypothesis & three-cell design
  2. Global agreement triangle (per axis)
  3. Headline figure: kappa-by-fame-quartile (Baseline vs ArmA, ArmA vs ArmB)
  4. Confusion-matrix heatmaps
  5. Fallback-rate panel (per arm per stratum)
  6. Methodology appendix (prompt diff, statistical caveats)

Usage:
    python scripts/build_directness_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
METRICS_PATH = ANALYSIS_DIR / "directness_metrics.json"
STRATIFIED_PATH = ANALYSIS_DIR / "stratified_metrics.csv"
FALLBACK_PATH = ANALYSIS_DIR / "fallback_rates.csv"
DASH_OUT = ANALYSIS_DIR / "directness_dashboard.html"

QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def _agreement_triangle_table(metrics: dict) -> str:
    """Render the global agreement triangle as an HTML table."""
    rows = []
    for axis, pairs in metrics.get("global", {}).items():
        for pair_name, m in pairs.items():
            rows.append({
                "axis": axis,
                "pair": pair_name.replace("__vs__", " vs "),
                "n": m.get("n"),
                "agreement": m.get("agreement"),
                "kappa": m.get("kappa"),
                "test": m.get("test_name") or "",
                "p_value": m.get("p_value"),
            })
    df = pd.DataFrame(rows)
    return df.to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.4f}")


def _kappa_by_fame_chart(stratified: pd.DataFrame) -> str:
    """Headline figure: kappa across fame quartiles per pair, faceted by axis."""
    sub = stratified[stratified["stratum"] == "fame_quartile"].copy()
    if sub.empty:
        return "<p><em>No fame-quartile data available.</em></p>"

    sub["stratum_value"] = pd.Categorical(
        sub["stratum_value"], categories=QUARTILE_ORDER, ordered=True
    )
    sub = sub.sort_values(["axis", "pair", "stratum_value"])

    fig = px.line(
        sub, x="stratum_value", y="kappa", color="pair", facet_col="axis",
        markers=True, hover_data=["n", "agreement", "p_value"],
        title="Cohen's kappa across fame quartiles, by axis and pair",
        labels={"stratum_value": "Fame quartile (Q1=obscure, Q4=famous)", "kappa": "Cohen's kappa"},
    )
    fig.update_yaxes(range=[-0.1, 1.05])
    fig.update_layout(height=480, legend_title_text="Pair")
    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="kappa-fame")


def _confusion_heatmaps(metrics: dict) -> str:
    """Render confusion matrices for the subclass axis as heatmaps."""
    pieces: list[str] = []
    for s1, s2 in [("baseline", "arm_a"), ("baseline", "arm_b"), ("arm_a", "arm_b")]:
        for axis in ("ai_native", "subclass", "rad_score", "cohort"):
            cm_path = ANALYSIS_DIR / f"confusion_{s1}_vs_{s2}__{axis}.csv"
            if not cm_path.exists():
                continue
            cm = pd.read_csv(cm_path, index_col=0)
            fig = go.Figure(data=go.Heatmap(
                z=cm.values, x=list(cm.columns), y=list(cm.index),
                colorscale="Blues", text=cm.values, texttemplate="%{text}",
                hovertemplate=f"{s1}=%{{y}}<br>{s2}=%{{x}}<br>n=%{{z}}<extra></extra>",
            ))
            fig.update_layout(
                title=f"{axis} | rows={s1}, cols={s2}",
                height=320, margin=dict(l=40, r=40, t=40, b=40),
            )
            pieces.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return "\n".join(pieces) if pieces else "<p><em>No confusion matrices found.</em></p>"


def _fallback_chart(fallback: pd.DataFrame) -> str:
    """Bar chart of fallback rate per arm per stratum."""
    if fallback.empty:
        return "<p><em>No fallback data.</em></p>"
    fig = px.bar(
        fallback, x="stratum_value", y="fallback_rate",
        color="arm", barmode="group", facet_col="stratum",
        hover_data=["n"],
        title="Insufficient-information fallback rate by stratum and arm",
        labels={"fallback_rate": "Fallback rate", "stratum_value": "Stratum value"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=420)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="fallback")


def main() -> None:
    if not METRICS_PATH.exists():
        print(f"ERROR: {METRICS_PATH} missing. Run analyze_directness.py first.", file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(METRICS_PATH.read_text())
    stratified = pd.read_csv(STRATIFIED_PATH) if STRATIFIED_PATH.exists() else pd.DataFrame()
    fallback = pd.read_csv(FALLBACK_PATH) if FALLBACK_PATH.exists() else pd.DataFrame()

    triangle = _agreement_triangle_table(metrics)
    kappa_chart = _kappa_by_fame_chart(stratified)
    cm_charts = _confusion_heatmaps(metrics)
    fallback_chart = _fallback_chart(fallback)

    n_total = metrics.get("n_total", 0)
    fb = metrics.get("fallback_rates_by_arm", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Directness Audit Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
          max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
  h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h2 {{ margin-top: 2em; }}
  .hero {{ background: #f8f9fa; padding: 1em 1.4em; border-left: 4px solid #4c6ef5; margin: 1em 0; }}
  table.table {{ border-collapse: collapse; margin: 1em 0; }}
  table.table th, table.table td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
  table.table th {{ background: #f1f3f5; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1em; }}
</style>
</head>
<body>
<h1>LLM Directness Audit</h1>
<p class="meta">Joined N = {n_total:,} companies (intersection of v2 baseline + Arm A + Arm B).</p>

<div class="hero">
<strong>Hypothesis.</strong> Does the LLM's v2 taxonomy verdict track the input
features (descriptions, keywords) it sees, or does it leak from memorized facts
about the company absorbed during pretraining? We compare the v2 baseline to two
ablations: <strong>Arm A</strong> strips inputs to real name + address, while
<strong>Arm B</strong> additionally replaces the name with a deterministic
anonymized token. Divergent kappa across fame quartiles between the two
ablations is the memorization signal.
</div>

<h2>1. Global agreement triangle</h2>
{triangle}

<h2>2. Headline: kappa-by-fame-quartile</h2>
<p class="meta">Compare the slope of <em>baseline vs arm_a</em> against
<em>arm_a vs arm_b</em> across Q1 (most obscure) -> Q4 (most famous). The two
slopes diverging at high fame is the leakage fingerprint.</p>
{kappa_chart}

<h2>3. Fallback rates per arm per stratum</h2>
<p class="meta">An arm that returns the <em>Insufficient information</em> fallback
(subclass=0A AND reasons containing "Insufficient information") for many obscure
companies but few famous ones is corroborating evidence that minimal-input judgments
ride on memorized identity.</p>
<p class="meta">Global rates: baseline {fb.get('baseline'):.3%},
arm_a {fb.get('arm_a'):.3%}, arm_b {fb.get('arm_b'):.3%}.</p>
{fallback_chart}

<h2>4. Confusion matrices</h2>
<div class="grid">
{cm_charts}
</div>

<h2>5. Methodology</h2>
<ul>
  <li>Both ablation arms use <code>prompts/directness_prompt.txt</code>, identical to
      <code>Multiclassification_prompt.txt</code> in the parent v2 repo except for the
      <em>Insufficient information</em> rule, which is relaxed to demand a best-effort
      judgment from minimal input. This is a known confound for the
      <em>baseline vs arm_a</em> comparison; the <em>arm_a vs arm_b</em> comparison
      isolates the name-identity signal because both arms share the relaxed prompt.</li>
  <li>Cohen's kappa is computed via <code>sklearn.metrics.cohen_kappa_score</code>
      with the canonical label set per axis. Multi-class axes use unweighted kappa.</li>
  <li>McNemar's test is used for binary axes (ai_native, cohort); Stuart-Maxwell
      marginal homogeneity test is used for multi-class axes (subclass, rad_score).</li>
  <li>Address-only signal can encode sector via geography (Sand Hill Road -> VC,
      Cambridge MA -> biotech). Quartile stratification partially addresses this; the
      strongest robustness check would hold city or zip3 fixed within a class.</li>
</ul>

<p class="meta">Generated from <code>outputs/analysis/directness_metrics.json</code>.</p>
</body>
</html>
"""
    DASH_OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {DASH_OUT}")


if __name__ == "__main__":
    main()
