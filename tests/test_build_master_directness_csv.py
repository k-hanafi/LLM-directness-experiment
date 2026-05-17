"""Integration tests for scripts/build_master_directness_csv.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_master_directness_csv.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_master_directness_csv", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_row(**overrides):
    base = {
        "org_uuid": "uuid-1",
        "name": "TestCo",
        "address": "1 Main",
        "city": "Boston",
        "state_code": "MA",
        "postal_code": "02101",
        "short_description": "Short",
        "category_list": "AI",
        "category_groups_list": "Software",
        "rank": 10,
        "total_funding_usd": 1_000_000,
        "num_funding_rounds": 2,
        "homepage_url": "https://example.com",
        "linkedin_url": "",
        "twitter_url": "",
        "facebook_url": "",
        "founded_date": "15jun2018",
        "year_founded": "",
        "description": "Long body",
    }
    base.update(overrides)
    return base


def test_build_script_produces_master_columns(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame([_minimal_row()]).to_csv(inp, index=False)
    subprocess.check_call(
        [sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(out), "--chunksize", "100"],
        cwd=PROJECT_ROOT,
    )
    mod = _load_build_module()
    got = pd.read_csv(out)
    assert list(got.columns) == list(mod.MASTER_OUTPUT_COLUMNS)
    assert got.loc[0, "founded_month_year"] == "Jun 2018"
    assert got.loc[0, "long_description"] == "Long body"
    assert "founded_date" not in got.columns
    assert "year_founded" not in got.columns


def test_year_only_founded_fallback(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    row = _minimal_row()
    row["founded_date"] = ""
    row["year_founded"] = 2019
    pd.DataFrame([row]).to_csv(inp, index=False)
    subprocess.check_call(
        [sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(out)],
        cwd=PROJECT_ROOT,
    )
    got = pd.read_csv(out)
    assert got.loc[0, "founded_month_year"] == "Jan 2019"


def test_missing_required_column_exits_nonzero(tmp_path: Path) -> None:
    inp = tmp_path / "bad.csv"
    pd.DataFrame([{"org_uuid": "x"}]).to_csv(inp, index=False)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(tmp_path / "o.csv")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "missing required" in proc.stderr.lower()
