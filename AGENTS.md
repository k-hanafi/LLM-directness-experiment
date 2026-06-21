# AGENTS.md

Briefing for AI coding agents (Cursor, Claude Code, Codex, cloud agents). **Read this before making changes, and update it as the project evolves** (see [Maintaining this file](#maintaining-this-file)).

## Project overview

A controlled experiment testing whether an LLM startup classifier (`gpt-5.4-nano`) derives its verdicts from the **input features it is shown** or from **facts memorized during pretraining**. It is a reliability audit spun off from the AI-native-startup classifier in Bena, Bian & Giannetti (2026), *Prompted to Start*. "Directness" is from Asirvatham, Mokski & Shleifer (2026), *GPT as a Measurement Tool*: a measurement is direct when driven by signal in the content, not leakage/memorized facts/shortcut cues.

**Design — three cells, one pipeline, one model snapshot, one prompt body.** The *only* experimentally-varied factor is which input fields are populated:

| Arm | CLI `--arm` | CompanyName | Descriptions / Keywords / Year | Address |
|-----|-------------|-------------|--------------------------------|---------|
| **Baseline** | `baseline` | real | real | real |
| **Arm A** (feature ablation) | `a` | real | `[not available]` | real |
| **Arm B** (identity ablation) | `b` | anonymized `Company-<hex>` | `[not available]` | real |

Baseline→A isolates whether descriptions do the work; A→B isolates whether the real name alone leaks signal via memorization.

## Status

All experimental data is generated; work is now in the **analysis & presentation** phase.

| Phase | State |
|-------|-------|
| Master dataset (~269k companies) | done — `data/master_csv_directness_experiment.csv` |
| Classification, all 3 arms | done — `outputs/<arm>/classified_<arm>.csv` |
| Fame proxy + quartiles | done — `outputs/analysis/fame_quartiles.csv` |
| Directness metrics (kappa, McNemar, Stuart-Maxwell) | done — `outputs/analysis/directness_metrics.json` |
| Ground-truth validation (Tavily, ~22k evidence subset) | done — `outputs/analysis/ground_truth_validation_metrics.json` |
| Dashboards (Plotly HTML) | done — see `data visualization/` |

## Tech stack

- **Python ≥ 3.11**. Deps in `pyproject.toml` / `requirements.txt`.
- **OpenAI Responses API** (`POST /v1/responses`) + **Batch API** for the 269k-row runs; sync Responses calls for `classify.py test`.
- **pydantic v2** (output schema + JSON-schema generation), **pandas/numpy** (data), **tiktoken** (cost preflight), **tenacity** (retry), **rich** (CLI/progress).
- **scikit-learn / scipy / statsmodels** (kappa, McNemar, Stuart-Maxwell, Wilcoxon), **plotly** (dashboards).
- **pytest** for tests. No CI configured yet.

## Repository layout

```
classify.py              # CLI entry point: prepare/submit/status/download/retry/merge/test/run
prompts/                 # arm_a_prompt.txt, arm_b_prompt.txt, baseline_prompt.txt (bodies identical except INPUT FORMAT)
src/                     # pipeline internals (see table below)
scripts/                 # dataset build + statistical analysis + dashboard
tests/                   # control-enforcing tests (prompt consistency, formatter per arm, schema, anonymizer)
data visualization/      # presentation HTML + standalone dashboard builders (01_Presentation_Materials, 02_Analysis_Code)
data/        (gitignored) # company_us_all_var_Khaled.csv (raw), master_csv_directness_experiment.csv (built)
outputs/     (gitignored) # baseline/ arm_a/ arm_b/ (batch artifacts + classified CSVs), analysis/, ground_truth/
keys/        (gitignored) # openai.env -> OPENAI_API_KEY (loaded by src/openai_config.py)
```

### `src/` module map

| Module | Responsibility |
|--------|----------------|
| `context.py` | Active-arm context; routes every output path under `outputs/<arm>/`. `set_active_arm()` is called by `classify.py` before any state-touching import. |
| `schema.py` | `ClassificationResult` (pydantic) — **single source of truth** for the 12-field output; auto-generates the JSON schema injected into requests. |
| `formatter.py` | Arm-aware user-message builder. **The experimental manipulation lives here** — which fields become `[not available]`. |
| `name_anonymizer.py` | Deterministic, leak-free Arm-B name: `Company-<8 hex of sha256(org_uuid)>`. |
| `builder.py` | Builds JSONL batch files (one Responses request per line, shared cached prefix). |
| `submitter.py` | Fault-tolerant upload + batch creation (tenacity backoff); `BillingLimitError`. |
| `monitor.py` | Async concurrent batch monitor with sliding-window queue-pressure control. |
| `downloader.py` | Downloads results/errors; aggregates `cached_tokens` for cost reporting. |
| `merger.py` | Merges per-batch parsed CSVs into `classified_<arm>.csv`. |
| `state.py` | `PipelineState`/`BatchRecord` JSON checkpoint per arm; atomic writes. |
| `tokens.py` | tiktoken cost estimate for `--dry-run`. |
| `openai_config.py` | Model default (`gpt-5.4-nano`), rate/batch limits, loads API key. |
| `logger.py` | Logging setup. |

### `scripts/` (run from repo root)

| Script | Purpose |
|--------|---------|
| `build_master_directness_csv.py` | Build column-minimal `master_csv_directness_experiment.csv` from the raw Khaled CSV. |
| `compute_fame_proxy.py` | Composite fame score → quartiles Q1 (obscure) … Q4 (famous). |
| `analyze_directness.py` | Headline stats: agreement, Cohen's kappa, McNemar (binary), Stuart-Maxwell (multi-class), stratified by fame. |
| `analyze_pretraining_reliability.py` | Validate Baseline & Arm A against Tavily evidence-grounded ground truth. |
| `build_directness_dashboard.py` | Render the in-repo Plotly directness dashboard (HTML). |

## Domain model — `ClassificationResult` (`src/schema.py`)

`CompanyID`, `CompanyName`, `ai_native` (0/1), `subclass` (`1A`–`1G`, `0A`–`0C`, `0`), `rad_score` (`RAD-H/M/L/NA`), `cohort` (`PRE-GENAI` / `GENAI-ERA`, split at GPT-4 / Mar 2023), `conf_classification` (1–5), `conf_rad` (1–5 or null), `reasons_3_points`, `sources_used`, `verification_critique`, `pretraining_inferred_description` (null for baseline).

## Pipeline / data flow

```
raw Khaled CSV
  └─ scripts/build_master_directness_csv.py ─→ data/master_csv_directness_experiment.csv
       └─ classify.py {prepare → submit → download → merge} --arm {baseline,a,b}
            ─→ outputs/<arm>/classified_<arm>.csv
                 ├─ scripts/compute_fame_proxy.py ─→ outputs/analysis/fame_quartiles.csv
                 ├─ scripts/analyze_directness.py ─→ outputs/analysis/*.json|csv
                 └─ scripts/analyze_pretraining_reliability.py ─→ ground-truth validation
                      └─ dashboards (scripts/ + data visualization/02_Analysis_Code/)
```

Per-arm pipeline stages: `prepared → submitted → in_progress → completed | failed | expired`, checkpointed in `outputs/<arm>/.../state.json`.

## Development commands

```bash
pip install -e ".[dev]"            # install with pytest
export OPENAI_API_KEY=...          # or put it in keys/openai.env

pytest                             # run all control tests

python classify.py run --arm baseline --dry-run     # cost preflight, no API calls
python classify.py run --arm a                      # full pipeline for one arm
python classify.py test --arm b --company-name Stripe   # classify one company synchronously
python classify.py status --arm a                   # batch progress

python scripts/build_master_directness_csv.py       # rebuild master dataset
python scripts/compute_fame_proxy.py                # then fame quartiles
python scripts/analyze_directness.py                # then headline metrics
```

## Where to work

| Task | Start here |
|------|-----------|
| Change what each arm shows the model | `src/formatter.py` (+ `prompts/*` INPUT FORMAT) |
| Change classification taxonomy / few-shot | all three `prompts/*.txt` **identically** outside INPUT FORMAT |
| Change output fields | `src/schema.py` (schema auto-propagates) |
| Add/adjust a CLI subcommand | `classify.py` |
| Batch/cost/rate-limit tuning | `src/openai_config.py`, `src/builder.py`, `src/monitor.py` |
| Anonymization scheme | `src/name_anonymizer.py` (+ `tests/test_name_anonymizer.py`) |
| New statistic / stratification | `scripts/analyze_directness.py` |
| Ground-truth validation | `scripts/analyze_pretraining_reliability.py` |
| Dashboards | `scripts/build_directness_dashboard.py`, `data visualization/02_Analysis_Code/` |

## Conventions & guardrails (do not break the experiment)

- **Prompt invariance is the core control.** The three `prompts/*.txt` must be byte-identical outside their INPUT FORMAT block. `tests/test_prompt_consistency.py` enforces this — run `pytest` after any prompt edit.
- **One varied factor only.** Anything affecting all arms (model, schema, taxonomy, scoring) must change *identically* across arms. The Address line is intentionally present in every arm.
- **Schema is generated, never hand-duplicated.** Edit `src/schema.py`; never write the JSON schema by hand elsewhere.
- **Arm B names stay deterministic & leak-free** (stable per `org_uuid`, no original token recoverable).
- **Paths are arm-aware** via `src/context.py`; never hardcode `outputs/...`. Call `set_active_arm()` before importing state-touching modules.
- **Secrets & big data stay out of git.** `keys/`, `data/`, `outputs/` are gitignored. Don't commit CSVs or API keys.
- Match existing style; keep changes surgical.

## Maintaining this file

This file is auto-loaded into agent context, so keeping it current keeps every future chat sharp. **Update it in the same change as the code**, no need to be asked, when you:

- finish or start a phase (update **Status**)
- add/remove/rename a top-level module, script, or prompt (update the **layout / module map**)
- change the pipeline, data flow, or domain model (`ClassificationResult`)
- change dev commands, dependencies, or add CI
- move or add an entry point

Keep edits surgical (~100–200 lines total); don't log session chatter or trivial fixes. When unsure, at minimum refresh **Status** and the **file map**. Global policy: `~/.cursor/user-rules/agents-md-maintenance.md`.
