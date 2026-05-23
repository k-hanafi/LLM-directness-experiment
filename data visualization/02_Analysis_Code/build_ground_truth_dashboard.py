#!/usr/bin/env python3
"""Build the ground-truth validation dashboard.

Reads the analysis outputs produced by scripts/analyze_pretraining_reliability.py
and writes a fully self-contained HTML dashboard to:
    data visualization/01_Presentation_Materials/ground_truth_validation_dashboard.html

Plotly is embedded inline at build time so the file works offline when opened
directly (file://) or emailed as a single attachment.

Usage:
    python "data visualization/02_Analysis_Code/build_ground_truth_dashboard.py"
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_DIR = _PROJECT_ROOT / "outputs" / "analysis"

OUTPUT_PATH = (
    _PROJECT_ROOT
    / "data visualization"
    / "01_Presentation_Materials"
    / "ground_truth_validation_dashboard.html"
)

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CACHE = Path(__file__).resolve().parent / ".cache" / "plotly-2.35.2.min.js"


def load_plotly_js() -> str:
    """Fetch Plotly once, cache locally, return minified JS for inline embedding."""
    PLOTLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not PLOTLY_CACHE.exists():
        print(f"  Downloading Plotly from {PLOTLY_URL} ...")
        urllib.request.urlretrieve(PLOTLY_URL, PLOTLY_CACHE)
    return PLOTLY_CACHE.read_text(encoding="utf-8")


def load_data() -> dict:
    with open(_ANALYSIS_DIR / "ground_truth_validation_metrics.json") as f:
        metrics = json.load(f)

    axis_df = pd.read_csv(_ANALYSIS_DIR / "ground_truth_agreement_by_axis.csv")
    conf_df = pd.read_csv(_ANALYSIS_DIR / "ground_truth_confidence_comparison.csv")
    tier_df = pd.read_csv(_ANALYSIS_DIR / "ground_truth_agreement_by_conf_tier.csv")
    base_df = pd.read_csv(_ANALYSIS_DIR / "ground_truth_base_rate_decomposition.csv")
    fame_df = pd.read_csv(_ANALYSIS_DIR / "ground_truth_agreement_by_fame.csv")

    axes = ["ai_native", "subclass", "rad_score", "cohort"]
    axis_labels = {
        "ai_native": "AI-Native (binary)",
        "subclass": "Subclass (10-class)",
        "rad_score": "RAD Score (4-class)",
        "cohort": "Cohort (binary)",
    }

    agreement_by_axis = {
        "axes": axes,
        "labels": [axis_labels[a] for a in axes],
        "baseline": [],
        "arm_a": [],
        "baseline_kappa": [],
        "arm_a_kappa": [],
    }
    for ax in axes:
        b_row = axis_df[(axis_df["pair"] == "baseline__vs__tavily") & (axis_df["axis"] == ax)]
        a_row = axis_df[(axis_df["pair"] == "arm_a__vs__tavily") & (axis_df["axis"] == ax)]
        agreement_by_axis["baseline"].append(round(float(b_row["agreement"].iloc[0]) * 100, 1) if len(b_row) else None)
        agreement_by_axis["arm_a"].append(round(float(a_row["agreement"].iloc[0]) * 100, 1) if len(a_row) else None)
        agreement_by_axis["baseline_kappa"].append(round(float(b_row["kappa"].iloc[0]), 3) if len(b_row) else None)
        agreement_by_axis["arm_a_kappa"].append(round(float(a_row["kappa"].iloc[0]), 3) if len(a_row) else None)

    conf_sources = {}
    for _, row in conf_df.iterrows():
        src = row["source"]
        if src in ("baseline", "arm_a", "tavily"):
            conf_sources[src] = {
                "mean": round(float(row["mean"]), 3),
                "median": float(row["median"]),
                "dist": [
                    round(float(row["pct_conf_1"]) * 100, 1),
                    round(float(row["pct_conf_2"]) * 100, 1),
                    round(float(row["pct_conf_3"]) * 100, 1),
                    round(float(row["pct_conf_4"]) * 100, 1),
                    round(float(row["pct_conf_5"]) * 100, 1),
                ],
            }

    wilcoxon = {}
    for _, row in conf_df.iterrows():
        if row["source"].startswith("wilcoxon_"):
            key = row["source"].replace("wilcoxon_", "")
            wilcoxon[key] = {
                "mean_delta": round(float(row["mean"]), 3),
                "p": float(row["wilcoxon_p"]) if pd.notna(row.get("wilcoxon_p")) else None,
            }

    base_rate = {}
    for _, row in base_df.iterrows():
        arm = row["arm"]
        if arm not in base_rate:
            base_rate[arm] = {}
        base_rate[arm][row["gt_class"]] = {
            "agreement": round(float(row["agreement"]) * 100, 1),
            "n": int(row["n"]),
            "interpretation": row["interpretation"],
        }

    fame_data = {
        "quartiles": ["Q1", "Q2", "Q3", "Q4"],
        "baseline_agreement": [],
        "arm_a_agreement": [],
        "baseline_kappa": [],
        "arm_a_kappa": [],
    }
    for q in fame_data["quartiles"]:
        b = fame_df[(fame_df["fame_quartile"] == q) & (fame_df["arm"] == "baseline") & (fame_df["axis"] == "ai_native")]
        a = fame_df[(fame_df["fame_quartile"] == q) & (fame_df["arm"] == "arm_a") & (fame_df["axis"] == "ai_native")]
        fame_data["baseline_agreement"].append(round(float(b["agreement"].iloc[0]) * 100, 1) if len(b) else None)
        fame_data["arm_a_agreement"].append(round(float(a["agreement"].iloc[0]) * 100, 1) if len(a) else None)
        fame_data["baseline_kappa"].append(round(float(b["kappa"].iloc[0]), 3) if len(b) else None)
        fame_data["arm_a_kappa"].append(round(float(a["kappa"].iloc[0]), 3) if len(a) else None)

    tier_data = {"baseline": {}, "arm_a": {}}
    for _, row in tier_df.iterrows():
        arm = row["arm"]
        axis = row["axis"]
        tier = int(row["conf_tier"])
        if axis not in tier_data[arm]:
            tier_data[arm][axis] = {}
        tier_data[arm][axis][tier] = round(float(row["agreement"]) * 100, 1)

    return {
        "n_total": metrics["n_total"],
        "accuracy_gap_pp": round(metrics["accuracy_gap_ai_native"] * 100, 1),
        "baseline_agreement": round(
            metrics["global"]["ai_native"]["baseline__vs__tavily"]["agreement"] * 100, 1
        ),
        "arm_a_agreement": round(
            metrics["global"]["ai_native"]["arm_a__vs__tavily"]["agreement"] * 100, 1
        ),
        "baseline_kappa": metrics["global"]["ai_native"]["baseline__vs__tavily"]["kappa"],
        "arm_a_kappa": metrics["global"]["ai_native"]["arm_a__vs__tavily"]["kappa"],
        "agreement_by_axis": agreement_by_axis,
        "conf_sources": conf_sources,
        "wilcoxon": wilcoxon,
        "base_rate": base_rate,
        "fame": fame_data,
        "tier": tier_data,
    }


def build_html(d: dict) -> str:
    data_json = json.dumps(d)
    arm_a_delta = d["wilcoxon"].get("arm_a_vs_tavily", {}).get("mean_delta", "2.70")
    baseline_delta = d["wilcoxon"].get("baseline_vs_tavily", {}).get("mean_delta", "0.83")

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pretraining Reliability Validation Dashboard</title>
<script>
__PLOTLY__
</script>
<style>
:root {{
  --bg: #ffffff;
  --bg2: #ffffff;
  --bg3: #f8f9fb;
  --border: #e5e7eb;
  --border2: #d1d5db;
  --text: #1a1a1a;
  --text2: #4a4a4a;
  --muted: #8a8a8a;
  --navy: #1e2a4a;
  --navy-light: #f0f2f7;
  --indigo: #4f46e5;
  --indigo-light: #eef2ff;
  --indigo-border: #c7d2fe;
  --emerald: #059669;
  --emerald-light: #ecfdf5;
  --emerald-border: #a7f3d0;
  --amber: #d97706;
  --amber-light: #fffbeb;
  --amber-border: #fde68a;
  --rose: #e11d48;
  --rose-light: #fff1f2;
  --rose-border: #fecdd3;
  --cyan: #0891b2;
  --violet: #7c3aed;
  --slate: #475569;
  --serif: Georgia, 'Times New Roman', serif;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  --mono: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.7; font-size: 15px; }}
::selection {{ background: var(--navy); color: white; }}

nav {{
  position: fixed; top: 0; left: 0; height: 100vh; width: 216px;
  padding: 2.25rem 1.75rem; background: #000000; border-right: 1px solid rgba(255,255,255,0.08);
  z-index: 100; display: flex; flex-direction: column; overflow-y: auto;
}}
.nav-brand {{ font-family: var(--serif); font-size: 1rem; font-weight: 600; color: #ffffff; margin-bottom: 0.2rem; }}
.nav-sub {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.5); margin-bottom: 2.5rem; }}
.nav-section {{ margin-bottom: 1.5rem; }}
.nav-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,255,255,0.4); margin-bottom: 0.55rem; }}
nav ul {{ list-style: none; }}
nav ul li {{ margin-bottom: 0.3rem; }}
nav ul a {{ color: rgba(255,255,255,0.65); text-decoration: none; font-size: 0.8rem; display: block; padding: 0.15rem 0; transition: color 0.15s; }}
nav ul a:hover, nav ul a.active {{ color: #ffffff; font-weight: 500; }}
.nav-meta {{ margin-top: auto; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); }}
.nav-meta p {{ font-size: 0.7rem; color: rgba(255,255,255,0.45); line-height: 1.6; }}
.nav-meta strong {{ color: rgba(255,255,255,0.7); }}

main {{ margin-left: 216px; }}
section {{
  padding: 5rem 4.5rem; max-width: 1100px;
  border-bottom: 1px solid var(--border);
  opacity: 0; transform: translateY(20px);
  transition: opacity 0.65s ease, transform 0.65s ease;
}}
section.visible {{ opacity: 1; transform: translateY(0); }}
section:last-of-type {{ border-bottom: none; }}

.section-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--navy); font-weight: 600; margin-bottom: 0.85rem; display: block; }}
h1 {{ font-family: var(--serif); font-size: clamp(2.2rem, 4vw, 3rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.4rem; color: var(--navy); }}
h2 {{ font-family: var(--serif); font-size: clamp(1.6rem, 2.8vw, 2rem); font-weight: 400; line-height: 1.2; margin-bottom: 0.85rem; color: var(--navy); }}
h3 {{ font-family: var(--serif); font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--navy); }}
p {{ color: var(--text2); font-size: 0.9rem; max-width: 720px; margin-bottom: 1.1rem; line-height: 1.75; }}
p:last-child {{ margin-bottom: 0; }}
code {{ font-family: var(--mono); font-size: 0.82em; background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; color: var(--navy); }}

.hero-metrics {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem;
  margin: 2.5rem 0 1.5rem;
}}
.metric-card {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.5rem 1.6rem;
  transition: box-shadow 0.2s;
}}
.metric-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.06); }}
.metric-card.emerald {{ border-color: var(--emerald-border); background: var(--emerald-light); }}
.metric-card.amber {{ border-color: var(--amber-border); background: var(--amber-light); }}
.metric-card.indigo {{ border-color: var(--indigo-border); background: var(--indigo-light); }}
.metric-card.rose {{ border-color: var(--rose-border); background: var(--rose-light); }}
.mc-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem; font-weight: 600; }}
.mc-val {{ font-family: var(--serif); font-size: 2.8rem; line-height: 1; margin-bottom: 0.4rem; color: var(--navy); }}
.metric-card.emerald .mc-val {{ color: var(--emerald); }}
.metric-card.amber .mc-val {{ color: var(--amber); }}
.metric-card.indigo .mc-val {{ color: var(--indigo); }}
.metric-card.rose .mc-val {{ color: var(--rose); }}
.mc-sub {{ font-size: 0.78rem; color: var(--muted); margin-bottom: 0.2rem; }}
.mc-kappa {{ font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }}

.stat-row {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 2rem 0;
}}
.stat-card {{
  background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.1rem 1.3rem;
}}
.stat-val {{ font-family: var(--mono); font-size: 1.35rem; font-weight: 500; color: var(--navy); }}
.stat-label {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.25rem; }}

.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0; }}
.chart-row.single {{ grid-template-columns: 1fr; }}
.chart-box {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden;
}}
.chart-box-header {{ padding: 1.1rem 1.4rem 0.75rem; border-bottom: 1px solid var(--border); }}
.chart-box-title {{ font-family: var(--serif); font-size: 1.1rem; font-weight: 600; color: var(--navy); margin-bottom: 0.3rem; }}
.chart-box-desc {{ font-size: 0.78rem; color: var(--muted); line-height: 1.5; }}
.chart-body {{ padding: 0.5rem; }}

.insight {{ padding: 1.2rem 1.5rem; border-radius: 10px; margin: 1.5rem 0; font-size: 0.85rem; line-height: 1.75; }}
.insight p {{ font-size: 0.85rem; max-width: none; margin-bottom: 0.4rem; }}
.insight p:last-child {{ margin-bottom: 0; }}
.insight ul {{ margin: 0.5rem 0 0 1.1rem; padding: 0; }}
.insight li {{ font-size: 0.85rem; max-width: none; margin-bottom: 0.35rem; color: var(--text2); line-height: 1.65; }}
.insight li:last-child {{ margin-bottom: 0; }}
.insight-blue {{ background: var(--indigo-light); border: 1px solid var(--indigo-border); color: var(--text2); }}
.insight-blue strong {{ color: #3730a3; }}
.insight-emerald {{ background: var(--emerald-light); border: 1px solid var(--emerald-border); color: var(--text2); }}
.insight-emerald strong {{ color: #065f46; }}
.insight-amber {{ background: var(--amber-light); border: 1px solid var(--amber-border); color: var(--text2); }}
.insight-amber strong {{ color: #92400e; }}
.insight-rose {{ background: var(--rose-light); border: 1px solid var(--rose-border); color: var(--text2); }}
.insight-rose strong {{ color: #9f1239; }}

.prec-recall-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;
}}
.pr-panel {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden;
}}
.pr-header {{
  padding: 0.9rem 1.3rem;
  font-family: var(--serif); font-size: 1.05rem; font-weight: 600; color: var(--navy);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.6rem;
}}
.pr-badge {{
  font-family: var(--mono); font-size: 0.65rem; padding: 0.2rem 0.55rem;
  border-radius: 99px; font-weight: 500;
}}
.pr-badge.baseline {{ background: var(--indigo-light); color: var(--indigo); border: 1px solid var(--indigo-border); }}
.pr-badge.arm-a {{ background: var(--amber-light); color: var(--amber); border: 1px solid var(--amber-border); }}
.pr-rows {{ padding: 0.5rem 0; }}
.pr-row {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.65rem 1.3rem; border-bottom: 1px solid var(--border);
}}
.pr-row:last-child {{ border-bottom: none; }}
.pr-metric-name {{ font-size: 0.8rem; color: var(--text2); }}
.pr-metric-n {{ font-size: 0.72rem; color: var(--muted); font-family: var(--mono); }}
.pr-metric-val {{
  font-family: var(--mono); font-size: 1.1rem; font-weight: 500; color: var(--navy);
}}
.pr-metric-val.good {{ color: var(--emerald); }}
.pr-metric-val.warn {{ color: var(--amber); }}
.pr-metric-val.bad {{ color: var(--rose); }}

footer {{
  padding: 2.5rem 4.5rem; text-align: center; color: var(--muted);
  font-size: 0.75rem; border-top: 1px solid var(--border); margin-left: 216px;
  line-height: 1.8;
}}
footer strong {{ color: var(--text2); }}

@media (max-width: 1100px) {{
  nav {{ display: none; }} main {{ margin-left: 0; }} section {{ padding: 3rem 1.5rem; }}
  .hero-metrics, .stat-row, .chart-row, .prec-recall-grid {{ grid-template-columns: 1fr; }}
  footer {{ margin-left: 0; padding: 2rem 1.5rem; }}
}}
@media print {{
  section {{ opacity: 1 !important; transform: none !important; }}
  nav {{ display: none; }} main {{ margin-left: 0; }}
}}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">Pretraining</div>
  <div class="nav-sub">Reliability Validation</div>
  <div class="nav-section">
    <div class="nav-label">Sections</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#axes">Accuracy by Axis</a></li>
      <li><a href="#confidence">Confidence Rent</a></li>
      <li><a href="#precision">Precision &amp; Recall</a></li>
      <li><a href="#fame">Fame Stratification</a></li>
      <li><a href="#discrimination">Conf. Discrimination</a></li>
    </ul>
  </div>
  <div class="nav-meta">
    <p><strong>Sample</strong><br>{d["n_total"]:,} companies<br>evidence-only subset</p>
    <p style="margin-top:0.75rem;"><strong>Ground Truth</strong><br>Tavily-grounded<br>homepage crawl</p>
    <p style="margin-top:0.75rem;"><strong>Model</strong><br>gpt-5.4-nano</p>
  </div>
</nav>

<main>

<section id="overview">
  <span class="section-label">Experiment</span>
  <h1>How Reliable Are Classifications<br>Made Solely from Memory?</h1>
  <p>
    We validate two classification arms against an external ground truth: {d["n_total"]:,} startups whose
    AI-native status was determined using live website evidence from Tavily homepage crawls.
    <strong>Baseline</strong> classifies from Crunchbase text.
    <strong>Arm&nbsp;A</strong> classifies from the company name and address alone, relying entirely on the model&rsquo;s pretraining knowledge.
    The accuracy gap between them, measured against the same benchmark, quantifies the marginal contribution of input features.
  </p>

  <div class="hero-metrics">
    <div class="metric-card emerald">
      <div class="mc-label">Baseline Accuracy</div>
      <div class="mc-val">{d["baseline_agreement"]}%</div>
      <div class="mc-sub">vs evidence-grounded labels</div>
      <div class="mc-kappa">Cohen&rsquo;s &kappa; = {d["baseline_kappa"]}</div>
    </div>
    <div class="metric-card amber">
      <div class="mc-label">Arm A Accuracy</div>
      <div class="mc-val">{d["arm_a_agreement"]}%</div>
      <div class="mc-sub">pretraining knowledge only</div>
      <div class="mc-kappa">Cohen&rsquo;s &kappa; = {d["arm_a_kappa"]}</div>
    </div>
    <div class="metric-card indigo">
      <div class="mc-label">Accuracy Gap</div>
      <div class="mc-val">{d["accuracy_gap_pp"]}pp</div>
      <div class="mc-sub">Baseline &minus; Arm A</div>
      <div class="mc-kappa">directness signal vs ground truth</div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p>
      <strong>Interpretation:</strong>
      At 83.0%, pretraining-only classifications are surprisingly accurate. The model has internalized
      substantial knowledge about which companies are AI-native. But at 91.1%, providing Crunchbase
      descriptions adds a meaningful 8.1 percentage-point lift in accuracy. Input features do work
      the model cannot do from memory alone.
    </p>
    <p>
      Critically, raw agreement rates are inflated by the 85% base rate of class&nbsp;0 (not AI-native).
      The Cohen&rsquo;s &kappa; difference (&kappa;&nbsp;=&nbsp;0.576 vs &kappa;&nbsp;=&nbsp;0.323) reveals a larger
      chance-corrected gap: Baseline&rsquo;s classifications are nearly twice as well-calibrated with the
      ground truth as Arm&nbsp;A&rsquo;s on the substantive classification task.
    </p>
  </div>
</section>

<section id="axes">
  <span class="section-label">01. Agreement</span>
  <h2>Accuracy Across All Classification Axes</h2>
  <p>
    Agreement with ground truth measured on all four output axes: binary AI-native, 10-class subclass,
    4-class RAD score, and binary cohort. The gap between Baseline and Arm A varies substantially by axis.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Agreement Rate by Axis</div>
        <div class="chart-box-desc">Raw agreement with evidence-grounded ground truth. Baseline (Crunchbase text) vs Arm A (pretraining only)</div>
      </div>
      <div class="chart-body"><div id="chart-axes-agreement" style="height:420px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Cohen&rsquo;s &kappa; by Axis</div>
        <div class="chart-box-desc">Chance-corrected agreement; controls for class imbalance; values above 0.6 indicate substantial agreement</div>
      </div>
      <div class="chart-body"><div id="chart-axes-kappa" style="height:420px;"></div></div>
    </div>
  </div>

  <div class="insight insight-amber">
    <p>
      <strong>Cohort anomaly:</strong> Arm A achieves <em>higher</em> agreement with ground truth on the
      cohort axis (PRE-GENAI vs GENAI-ERA) than Baseline. Cohort is largely inferrable from a company&rsquo;s
      name and founding era, a signal the model can recover from pretraining without needing text descriptions.
      This is a rare axis where memorized knowledge is sufficient, and Baseline&rsquo;s longer text inputs appear to
      introduce noise rather than signal.
    </p>
  </div>
</section>

<section id="confidence">
  <span class="section-label">02. Information Rent</span>
  <h2>The Confidence Information Rent</h2>
  <p>
    Each step up in input richness (from pretraining memory alone, to Crunchbase text, to live website
    evidence) produces a measurable rise in the model&rsquo;s self-reported classification confidence.
    This is not a calibration artifact. Each prompt places the model in a different rational epistemic state:
    Arm&nbsp;A explicitly lacks input fields (low confidence is rational), Baseline has moderate text (moderate
    confidence is rational), and Tavily provides rich multi-page evidence (high confidence is rational).
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Mean Confidence by Input Richness</div>
        <div class="chart-box-desc">Three input-richness levels: Arm A (memory only) &rarr; Baseline (Crunchbase text) &rarr; Tavily GT (website evidence)</div>
      </div>
      <div class="chart-body"><div id="chart-conf-means" style="height:340px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Confidence Distribution by Source</div>
        <div class="chart-box-desc">% of classifications at each confidence level (1&ndash;5). Arm A floor-clusters at 1&ndash;2; Tavily ceiling-clusters at 4&ndash;5</div>
      </div>
      <div class="chart-body"><div id="chart-conf-dist" style="height:400px;"></div></div>
    </div>
  </div>

  <div class="stat-row">
    <div class="stat-card">
      <div class="stat-val">{d["conf_sources"]["arm_a"]["mean"]}</div>
      <div class="stat-label">Arm A mean conf</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{d["conf_sources"]["baseline"]["mean"]}</div>
      <div class="stat-label">Baseline mean conf</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{d["conf_sources"]["tavily"]["mean"]}</div>
      <div class="stat-label">Tavily GT mean conf</div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p>
      <strong>Wilcoxon signed-rank tests (paired, p&nbsp;&lt;&nbsp;0.001 for both):</strong>
      The shift from Arm&nbsp;A to Tavily is the larger gap
      (&Delta;&nbsp;=&nbsp;+{arm_a_delta}),
      while the shift from Baseline to Tavily is more modest
      (&Delta;&nbsp;=&nbsp;+{baseline_delta}).
      Website evidence buys disproportionately more certainty over pretraining memory than over
      Crunchbase text, confirming that the model finds descriptions moderately informative but
      homepage content substantially more diagnostic.
    </p>
  </div>
</section>

<section id="precision">
  <span class="section-label">03. Precision &amp; Recall</span>
  <h2>Where Arm A Breaks Down: Precision Collapse</h2>
  <p>
    Raw agreement rates are dominated by the 85% non-AI-native base rate. Decomposing into precision,
    recall, and specificity reveals the true failure mode of pretraining-only classification.
  </p>

  <div class="prec-recall-grid">
    <div class="pr-panel">
      <div class="pr-header">
        Baseline
        <span class="pr-badge baseline">Crunchbase text</span>
      </div>
      <div class="pr-rows">
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Recall (Sensitivity)</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["baseline"]["ai_native=1"]["n"]:,} GT-positive companies</div>
          </div>
          <div class="pr-metric-val warn">{d["base_rate"]["baseline"]["ai_native=1"]["agreement"]}%</div>
        </div>
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Specificity</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["baseline"]["ai_native=0"]["n"]:,} GT-negative companies</div>
          </div>
          <div class="pr-metric-val good">{d["base_rate"]["baseline"]["ai_native=0"]["agreement"]}%</div>
        </div>
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Precision (PPV)</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["baseline"]["arm_predicts_1"]["n"]:,} Baseline-positive predictions</div>
          </div>
          <div class="pr-metric-val good">{d["base_rate"]["baseline"]["arm_predicts_1"]["agreement"]}%</div>
        </div>
      </div>
    </div>

    <div class="pr-panel">
      <div class="pr-header">
        Arm A
        <span class="pr-badge arm-a">Pretraining only</span>
      </div>
      <div class="pr-rows">
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Recall (Sensitivity)</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["arm_a"]["ai_native=1"]["n"]:,} GT-positive companies</div>
          </div>
          <div class="pr-metric-val warn">{d["base_rate"]["arm_a"]["ai_native=1"]["agreement"]}%</div>
        </div>
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Specificity</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["arm_a"]["ai_native=0"]["n"]:,} GT-negative companies</div>
          </div>
          <div class="pr-metric-val warn">{d["base_rate"]["arm_a"]["ai_native=0"]["agreement"]}%</div>
        </div>
        <div class="pr-row">
          <div>
            <div class="pr-metric-name">Precision (PPV)</div>
            <div class="pr-metric-n">n&nbsp;=&nbsp;{d["base_rate"]["arm_a"]["arm_predicts_1"]["n"]:,} Arm A-positive predictions</div>
          </div>
          <div class="pr-metric-val bad">{d["base_rate"]["arm_a"]["arm_predicts_1"]["agreement"]}%</div>
        </div>
      </div>
    </div>
  </div>

  <div class="insight insight-rose">
    <p>
      <strong>Precision collapse is the key finding.</strong>
      When Arm&nbsp;A predicts a company is AI-native, it is correct only
      <strong>{d["base_rate"]["arm_a"]["arm_predicts_1"]["agreement"]}%</strong> of the time,
      compared to <strong>{d["base_rate"]["baseline"]["arm_predicts_1"]["agreement"]}%</strong> for Baseline.
      The model over-fires: it associates AI-sounding names with AI-native labels even when the company
      is not AI-native by the evidence-grounded standard. Pretraining memory is not merely incomplete.
      It is systematically biased toward false positives on the AI-native label.
    </p>
    <p>
      Recall is similarly lower for Arm&nbsp;A ({d["base_rate"]["arm_a"]["ai_native=1"]["agreement"]}% vs
      {d["base_rate"]["baseline"]["ai_native=1"]["agreement"]}%), but the 9.1pp specificity gap
      (90.9% vs 99.2%) shows Arm&nbsp;A also misclassifies genuinely non-AI-native companies at a higher rate,
      confirming bidirectional noise, not just conservative or aggressive bias.
    </p>
  </div>
</section>

<section id="fame">
  <span class="section-label">04. Stratification</span>
  <h2>Fame Stratification: Where Pretraining Leaks Most</h2>
  <p>
    Companies are split into fame quartiles Q1 (most obscure) through Q4 (most famous) using a composite
    proxy of funding, headcount, and web presence. If pretraining memorization drives Arm&nbsp;A&rsquo;s
    classifications, accuracy should be highest for well-known companies (Q4) and should fall as fame rises
    for Baseline (which should not rely on name recognition).
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">AI-Native Agreement by Fame Quartile</div>
        <div class="chart-box-desc">Q1 = most obscure, Q4 = most famous. The diverging lines are the signature of pretraining memorization</div>
      </div>
      <div class="chart-body"><div id="chart-fame-agreement" style="height:380px;"></div></div>
    </div>
  </div>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Cohen&rsquo;s &kappa; by Fame Quartile</div>
        <div class="chart-box-desc">Chance-corrected agreement: Baseline is relatively stable across quartiles; Arm A collapses at Q4</div>
      </div>
      <div class="chart-body"><div id="chart-fame-kappa" style="height:380px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p><strong>Possible explanations for falling agreement at Q4:</strong></p>
    <ul>
      <li><strong>Both arms:</strong> Famous firms are harder borderline cases (AI-augmented vs AI-native, multi-product giants).</li>
      <li><strong>Both arms:</strong> Crunchbase text and live website evidence may diverge more for well-known companies with heavy AI marketing.</li>
      <li><strong>Arm A:</strong> Name recognition triggers AI-native guesses on tech brands the model half-remembers.</li>
      <li><strong>Arm A:</strong> Memorized narratives may be stale or oversimplified relative to current homepage positioning.</li>
      <li><strong>Arm A:</strong> Partial recall can be worse than no recall: confident wrong labels instead of a conservative default.</li>
    </ul>
  </div>
</section>

<section id="discrimination">
  <span class="section-label">05. Discrimination</span>
  <h2>Confidence as a Reliability Signal Within Each Arm</h2>
  <p>
    Confidence scales are not comparable across arms (each prompt induces a different rational epistemic state).
    But within each arm, we can test whether higher self-reported confidence predicts higher accuracy against
    ground truth.
  </p>

  <div class="chart-row single">
    <div class="chart-box">
      <div class="chart-box-header">
        <div class="chart-box-title">Agreement Rate vs Confidence Tier (ai_native)</div>
        <div class="chart-box-desc">Within-arm discrimination: does the model know when it knows? Baseline: broadly monotonic. Arm A: non-monotonic dip at tier 3, then recovery.</div>
      </div>
      <div class="chart-body"><div id="chart-conf-discrim" style="height:400px;"></div></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <p>
      <strong>Arm A&rsquo;s high-confidence tail is remarkably accurate.</strong>
      Only 518 companies (~2.4%) receive conf&nbsp;&ge;&nbsp;4 from Arm&nbsp;A.
      Among conf&nbsp;=&nbsp;4 predictions: {d["tier"]["arm_a"].get("ai_native", {}).get(4, "92.8")}% agreement.
      Among conf&nbsp;=&nbsp;5: {d["tier"]["arm_a"].get("ai_native", {}).get(5, "100.0")}% agreement.
      When the model explicitly claims to recognize a company from memory, it is correct.
    </p>
    <p>
      The <strong>conf&nbsp;=&nbsp;3 dip</strong> ({d["tier"]["arm_a"].get("ai_native", {}).get(3, "64.1")}%)
      is the telling case: these are companies the model partially recalls but cannot confidently classify.
      Partial memorization is worse than no memorization. The model produces a confident-enough guess
      that bypasses the conservative default, but that guess is unreliable.
    </p>
  </div>
</section>

</main>

<footer>
  <strong>Experiment:</strong> LLM Directness: Ground Truth Validation &nbsp;&middot;&nbsp;
  <strong>Model:</strong> gpt-5.4-nano &nbsp;&middot;&nbsp;
  <strong>Ground Truth:</strong> Tavily homepage crawl, evidence-only subset &nbsp;&middot;&nbsp;
  <strong>n:</strong> {d["n_total"]:,} companies
  <br>
  Baseline = Crunchbase text input; Arm A = company name + address only (pretraining knowledge).
  Agreement measured against tavily-grounded classifications on companies with non-empty website evidence.
</footer>

<script>
const D = {data_json};

const COLORS = {{
  baseline: '#4f46e5',
  arm_a: '#d97706',
  tavily: '#059669',
  gap: '#e11d48',
}};

const plotlyConfig = {{displayModeBar: false, responsive: true}};
const axisFont = {{family: 'system-ui, -apple-system, sans-serif', size: 11, color: '#4a4a4a'}};
const titleFont = {{family: 'system-ui, -apple-system, sans-serif', size: 12, color: '#1e2a4a'}};

function baseLayout(extra) {{
  return Object.assign({{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: axisFont,
    xaxis: {{gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'}},
    yaxis: {{gridcolor: '#f0f0f0', zerolinecolor: '#e5e7eb'}},
    margin: {{l: 60, r: 24, t: 24, b: 50}},
    legend: {{orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: {{size: 11}}}},
  }}, extra || {{}});
}}

function renderAxesAgreement() {{
  const ax = D.agreement_by_axis;
  Plotly.newPlot('chart-axes-agreement', [
    {{
      type: 'bar', name: 'Baseline',
      x: ax.labels, y: ax.baseline,
      marker: {{color: COLORS.baseline}},
      text: ax.baseline.map(v => v + '%'),
      textposition: 'outside',
      textfont: {{family: 'ui-monospace, monospace', size: 10}},
      hovertemplate: 'Baseline<br>%{{x}}: %{{y}}%<extra></extra>',
    }},
    {{
      type: 'bar', name: 'Arm A',
      x: ax.labels, y: ax.arm_a,
      marker: {{color: COLORS.arm_a}},
      text: ax.arm_a.map(v => v + '%'),
      textposition: 'outside',
      textfont: {{family: 'ui-monospace, monospace', size: 10}},
      hovertemplate: 'Arm A<br>%{{x}}: %{{y}}%<extra></extra>',
    }},
  ], baseLayout({{
    barmode: 'group',
    yaxis: {{title: {{text: 'Agreement with Ground Truth (%)', font: titleFont}}, range: [0, 110]}},
    xaxis: {{tickfont: {{family: 'ui-monospace, monospace', size: 11}}}},
    margin: {{l: 65, r: 24, t: 30, b: 60}},
  }}), plotlyConfig);
}}

function renderAxesKappa() {{
  const ax = D.agreement_by_axis;
  Plotly.newPlot('chart-axes-kappa', [
    {{
      type: 'bar', name: 'Baseline',
      x: ax.labels, y: ax.baseline_kappa,
      marker: {{color: COLORS.baseline}},
      text: ax.baseline_kappa.map(v => v != null ? v.toFixed(3) : ''),
      textposition: 'outside',
      textfont: {{family: 'ui-monospace, monospace', size: 10}},
      hovertemplate: 'Baseline<br>%{{x}}: \u03ba = %{{y:.3f}}<extra></extra>',
    }},
    {{
      type: 'bar', name: 'Arm A',
      x: ax.labels, y: ax.arm_a_kappa,
      marker: {{color: COLORS.arm_a}},
      text: ax.arm_a_kappa.map(v => v != null ? v.toFixed(3) : ''),
      textposition: 'outside',
      textfont: {{family: 'ui-monospace, monospace', size: 10}},
      hovertemplate: 'Arm A<br>%{{x}}: \u03ba = %{{y:.3f}}<extra></extra>',
    }},
  ], baseLayout({{
    barmode: 'group',
    yaxis: {{title: {{text: "Cohen's κ", font: titleFont}}, range: [0, 0.8]}},
    xaxis: {{tickfont: {{family: 'ui-monospace, monospace', size: 11}}}},
    margin: {{l: 60, r: 24, t: 30, b: 60}},
    shapes: [{{
      type: 'line', x0: -0.5, x1: 3.5,
      y0: 0.6, y1: 0.6,
      line: {{color: '#059669', width: 1, dash: 'dot'}},
    }}],
    annotations: [{{
      x: 3.4, y: 0.63, text: 'Substantial (0.6)', showarrow: false,
      font: {{size: 9, color: '#059669', family: 'ui-monospace, monospace'}}, xanchor: 'right',
    }}],
  }}), plotlyConfig);
}}

function renderConfMeans() {{
  const sources = ['arm_a', 'baseline', 'tavily'];
  const labels = ['Arm A (memory only)', 'Baseline (Crunchbase text)', 'Tavily GT (website evidence)'];
  const means = sources.map(s => D.conf_sources[s].mean);
  const colors = [COLORS.arm_a, COLORS.baseline, COLORS.tavily];
  Plotly.newPlot('chart-conf-means', [{{
    type: 'bar',
    x: labels, y: means,
    marker: {{color: colors}},
    text: means.map(v => v.toFixed(3)),
    textposition: 'outside',
    textfont: {{family: 'ui-monospace, monospace', size: 12}},
    hovertemplate: '%{{x}}<br>Mean confidence: %{{y:.3f}}<extra></extra>',
    width: [0.45, 0.45, 0.45],
  }}], baseLayout({{
    yaxis: {{title: {{text: 'Mean conf_classification (1–5)', font: titleFont}}, range: [0, 5.5]}},
    xaxis: {{tickfont: {{family: 'ui-monospace, monospace', size: 11}}}},
    showlegend: false,
    margin: {{l: 70, r: 24, t: 20, b: 60}},
  }}), plotlyConfig);
}}

function renderConfDist() {{
  const sources = ['arm_a', 'baseline', 'tavily'];
  const srcLabels = ['Arm A', 'Baseline', 'Tavily GT'];
  const tierColors = ['#e11d48', '#fb7185', '#d97706', '#059669', '#065f46'];

  const traces = [1, 2, 3, 4, 5].map((tier, ti) => ({{
    type: 'bar',
    name: 'conf = ' + tier,
    x: srcLabels,
    y: sources.map(s => D.conf_sources[s].dist[ti]),
    marker: {{color: tierColors[ti]}},
    hovertemplate: 'conf = ' + tier + '<br>%{{x}}: %{{y:.1f}}%<extra></extra>',
  }}));
  Plotly.newPlot('chart-conf-dist', traces, baseLayout({{
    barmode: 'stack',
    yaxis: {{title: {{text: '% of classifications', font: titleFont}}, range: [0, 105]}},
    xaxis: {{tickfont: {{family: 'ui-monospace, monospace', size: 12}}}},
    legend: {{orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: {{size: 11}}}},
    margin: {{l: 65, r: 24, t: 40, b: 50}},
  }}), plotlyConfig);
}}

function renderFameAgreement() {{
  const f = D.fame;
  Plotly.newPlot('chart-fame-agreement', [
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Baseline',
      x: f.quartiles, y: f.baseline_agreement,
      line: {{color: COLORS.baseline, width: 2.5}},
      marker: {{size: 8, color: COLORS.baseline}},
      text: f.baseline_agreement.map(v => v + '%'),
      textposition: 'top center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.baseline}},
      hovertemplate: 'Baseline %{{x}}: %{{y}}%<extra></extra>',
    }},
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Arm A',
      x: f.quartiles, y: f.arm_a_agreement,
      line: {{color: COLORS.arm_a, width: 2.5}},
      marker: {{size: 8, color: COLORS.arm_a}},
      text: f.arm_a_agreement.map(v => v + '%'),
      textposition: 'bottom center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.arm_a}},
      hovertemplate: 'Arm A %{{x}}: %{{y}}%<extra></extra>',
    }},
  ], baseLayout({{
    yaxis: {{title: {{text: 'Agreement with Ground Truth (%)', font: titleFont}}, range: [65, 100]}},
    xaxis: {{title: {{text: 'Fame Quartile (Q1 = obscure → Q4 = famous)', font: titleFont}}, tickfont: {{family: 'ui-monospace, monospace', size: 12}}}},
    margin: {{l: 70, r: 24, t: 30, b: 60}},
  }}), plotlyConfig);
}}

function renderFameKappa() {{
  const f = D.fame;
  Plotly.newPlot('chart-fame-kappa', [
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Baseline',
      x: f.quartiles, y: f.baseline_kappa,
      line: {{color: COLORS.baseline, width: 2.5}},
      marker: {{size: 8, color: COLORS.baseline}},
      text: f.baseline_kappa.map(v => v != null ? v.toFixed(3) : ''),
      textposition: 'top center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.baseline}},
      hovertemplate: 'Baseline %{{x}}: \u03ba = %{{y:.3f}}<extra></extra>',
    }},
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Arm A',
      x: f.quartiles, y: f.arm_a_kappa,
      line: {{color: COLORS.arm_a, width: 2.5}},
      marker: {{size: 8, color: COLORS.arm_a}},
      text: f.arm_a_kappa.map(v => v != null ? v.toFixed(3) : ''),
      textposition: 'bottom center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.arm_a}},
      hovertemplate: 'Arm A %{{x}}: \u03ba = %{{y:.3f}}<extra></extra>',
    }},
  ], baseLayout({{
    yaxis: {{title: {{text: "Cohen's κ", font: titleFont}}, range: [0, 0.7]}},
    xaxis: {{title: {{text: 'Fame Quartile (Q1 = obscure → Q4 = famous)', font: titleFont}}, tickfont: {{family: 'ui-monospace, monospace', size: 12}}}},
    margin: {{l: 60, r: 24, t: 30, b: 60}},
  }}), plotlyConfig);
}}

function renderConfDiscrim() {{
  const tiers = [1, 2, 3, 4, 5];
  const bVals = tiers.map(t => D.tier.baseline.ai_native[t] || null);
  const aVals = tiers.map(t => D.tier.arm_a.ai_native[t] || null);
  Plotly.newPlot('chart-conf-discrim', [
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Baseline',
      x: tiers, y: bVals,
      line: {{color: COLORS.baseline, width: 2.5}},
      marker: {{size: 9, color: COLORS.baseline}},
      text: bVals.map(v => v != null ? v + '%' : ''),
      textposition: 'top center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.baseline}},
      hovertemplate: 'Baseline conf=%{{x}}: %{{y}}% agreement<extra></extra>',
    }},
    {{
      type: 'scatter', mode: 'lines+markers', name: 'Arm A',
      x: tiers, y: aVals,
      line: {{color: COLORS.arm_a, width: 2.5}},
      marker: {{size: 9, color: COLORS.arm_a}},
      text: aVals.map(v => v != null ? v + '%' : ''),
      textposition: 'bottom center',
      textfont: {{family: 'ui-monospace, monospace', size: 10, color: COLORS.arm_a}},
      hovertemplate: 'Arm A conf=%{{x}}: %{{y}}% agreement<extra></extra>',
    }},
  ], baseLayout({{
    yaxis: {{title: {{text: 'Agreement with Ground Truth (%)', font: titleFont}}, range: [50, 105]}},
    xaxis: {{
      title: {{text: 'Confidence Tier (self-reported, 1–5)', font: titleFont}},
      tickfont: {{family: 'ui-monospace, monospace', size: 12}},
      dtick: 1,
    }},
    margin: {{l: 70, r: 24, t: 30, b: 60}},
    shapes: [{{
      type: 'rect', x0: 3.5, x1: 5.5, y0: 50, y1: 105,
      fillcolor: 'rgba(5,150,105,0.05)', line: {{width: 0}},
    }}],
    annotations: [{{
      x: 4.5, y: 104, text: 'high-conf tail (Arm A)', showarrow: false,
      font: {{size: 9, color: '#059669', family: 'ui-monospace, monospace'}},
    }}],
  }}), plotlyConfig);
}}

renderAxesAgreement();
renderAxesKappa();
renderConfMeans();
renderConfDist();
renderFameAgreement();
renderFameKappa();
renderConfDiscrim();

const observer = new IntersectionObserver((entries) => {{
  entries.forEach(e => {{ if (e.isIntersecting) e.target.classList.add('visible'); }});
}}, {{ threshold: 0.06, rootMargin: '0px 0px -30px 0px' }});
document.querySelectorAll('section').forEach(s => observer.observe(s));
document.getElementById('overview').classList.add('visible');

const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('nav ul a');
const navObs = new IntersectionObserver((entries) => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      navLinks.forEach(a => a.classList.remove('active'));
      const link = document.querySelector('nav ul a[href="#' + e.target.id + '"]');
      if (link) link.classList.add('active');
    }}
  }});
}}, {{ threshold: 0.25 }});
sections.forEach(s => navObs.observe(s));
</script>

</body>
</html>'''


def main() -> None:
    print("Loading analysis data ...")
    d = load_data()
    print(f"  n = {d['n_total']:,}, gap = {d['accuracy_gap_pp']}pp")

    print("Building HTML ...")
    html = build_html(d)
    plotly_js = load_plotly_js().replace("</script>", "<\\/script>")
    html = html.replace("__PLOTLY__", plotly_js)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
