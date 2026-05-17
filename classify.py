#!/usr/bin/env python3
"""CLI entry point for the directness experiment pipeline.

Uses the OpenAI **Responses API** (`POST /v1/responses`) for batch jobs and for
`classify.py test` (structured JSON via `text.format`).

Three arms, all run through the same pipeline in this repo:
  --arm baseline   full inputs (descriptions, address, keywords, year)
  --arm a          minimal inputs: real CompanyName + Address only
  --arm b          minimal inputs: anonymized CompanyName + Address only

The same dataset, model snapshot, code path, and prompt body are used across
all three. The only experimentally-varied factor is which fields are populated
in the user message (and a one-paragraph INPUT FORMAT change in the prompt).

Usage:
    python classify.py prepare  --arm baseline [--dry-run]
    python classify.py prepare  --arm a        [--dry-run]
    python classify.py prepare  --arm b        [--dry-run]
    python classify.py submit   --arm <arm>    [--concurrency 1]
    python classify.py status   --arm <arm>
    python classify.py download --arm <arm>
    python classify.py retry    --arm <arm>
    python classify.py merge    --arm <arm>    [--output path]
    python classify.py test     --arm <arm>    --company-id <id>
    python classify.py run      --arm <arm>    [--dry-run] [--concurrency 1]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.context import (
    VALID_ARMS,
    project_root,
    set_active_arm,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_CSV = project_root() / "data" / "master_csv_directness_experiment.csv"


def _resolve_data(args: argparse.Namespace) -> Path:
    raw = getattr(args, "data", None)
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else project_root() / p
    return DEFAULT_DATA_CSV


# -- Subcommand handlers -------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> None:
    """Build JSONL batch files for the active arm. With --dry-run, print cost only."""
    from src.builder import build_batch_files, load_system_prompt
    from src.openai_config import ESTIMATED_TOKENS_PER_REQUEST
    from src.formatter import format_user_message
    from src.logger import setup_logging
    from src.state import BatchRecord, PipelineState
    from src.tokens import estimate_cost

    setup_logging()

    data_csv = _resolve_data(args)
    row_slice = _parse_rows(args.rows)
    df = pd.read_csv(data_csv)
    if row_slice is not None:
        df = df.iloc[row_slice]

    user_messages = [
        format_user_message(row._asdict(), arm=args.arm)
        for row in df.itertuples(index=False)
    ]
    system_prompt = load_system_prompt(args.arm)

    estimate = estimate_cost(system_prompt, user_messages, args.model, args.batch_size)
    print(estimate.format_report())
    print(f"  Arm:                {args.arm}")
    print(f"  Data CSV:           {data_csv}")
    arm_dirname = "baseline" if args.arm == "baseline" else f"arm_{args.arm}"
    print(f"  Output dir:         outputs/{arm_dirname}/")
    print("=" * 60)

    if args.dry_run:
        return

    files = build_batch_files(
        data_csv,
        arm=args.arm,
        model=args.model,
        batch_size=args.batch_size,
        row_slice=row_slice,
    )

    state = PipelineState.load()
    state.batches = {}
    state.run_id = ""
    state.model = args.model
    state.arm = args.arm
    state.total_companies = len(df)

    for idx, fpath in enumerate(files, start=1):
        row_start = (idx - 1) * args.batch_size
        row_end = min(row_start + args.batch_size - 1, len(df) - 1)
        key = fpath.stem
        state.batches[key] = BatchRecord(
            batch_number=idx,
            file_path=str(fpath),
            row_range=f"{row_start}-{row_end}",
            estimated_tokens=args.batch_size * ESTIMATED_TOKENS_PER_REQUEST,
        )

    state.save()
    logger.info(
        "Prepared %d batch files for arm=%s. Run 'classify.py submit --arm %s' next.",
        len(files), args.arm, args.arm,
    )


def _cmd_submit(args: argparse.Namespace) -> None:
    from src.logger import setup_logging
    from src.monitor import submit_and_monitor
    from src.state import PipelineState

    setup_logging()
    state = PipelineState.load()

    if not state.batches:
        logger.error(
            "No batches prepared for arm=%s. Run 'classify.py prepare --arm %s' first.",
            args.arm, args.arm,
        )
        sys.exit(1)

    submit_and_monitor(
        state,
        concurrency=args.concurrency,
        model=args.model,
        batch_size=args.batch_size,
    )


def _cmd_status(args: argparse.Namespace) -> None:
    from src.logger import setup_logging
    from src.monitor import print_status
    from src.state import PipelineState

    setup_logging()
    state = PipelineState.load()
    print_status(state)


def _cmd_download(args: argparse.Namespace) -> None:
    from src.downloader import download_completed
    from src.logger import setup_logging
    from src.state import PipelineState

    setup_logging()
    state = PipelineState.load()
    download_completed(state)


def _download_error_files(state) -> None:
    from src.context import batch_errors_dir
    from src.downloader import _download_file
    from src.submitter import get_client

    errors_dir = batch_errors_dir()
    batches_needing_download = [
        rec for rec in state.completed_batches() + state.failed_batches()
        if rec.error_file_id
        and not (errors_dir / f"batch_{rec.batch_number:04d}_errors.jsonl").exists()
    ]
    if not batches_needing_download:
        return

    errors_dir.mkdir(parents=True, exist_ok=True)
    client = get_client()
    for rec in batches_needing_download:
        dest = errors_dir / f"batch_{rec.batch_number:04d}_errors.jsonl"
        _download_file(client, rec.error_file_id, dest)


def _cmd_retry(args: argparse.Namespace) -> None:
    from src.openai_config import ESTIMATED_TOKENS_PER_REQUEST
    from src.context import batch_requests_dir
    from src.downloader import collect_failed_custom_ids
    from src.logger import setup_logging
    from src.state import BatchRecord, PipelineState

    setup_logging()
    import json

    state = PipelineState.load()

    logger.info("Downloading error files for batches with failures...")
    _download_error_files(state)

    failed_ids = collect_failed_custom_ids(state)
    if not failed_ids:
        logger.info("No failed requests to retry.")
        return

    failed_id_set = set(failed_ids)
    logger.info("Found %d unique failed requests.", len(failed_id_set))

    retry_lines: list[str] = []
    for rec in sorted(state.batches.values(), key=lambda b: b.batch_number):
        if rec.failed_count == 0 or rec.file_path.endswith("retry_batch.jsonl"):
            continue
        fpath = Path(rec.file_path)
        if not fpath.exists():
            logger.warning("Original batch file missing: %s", fpath)
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("custom_id") in failed_id_set:
                    retry_lines.append(line.rstrip("\n"))

    if not retry_lines:
        logger.warning("Could not extract any matching requests from original JSONL files.")
        return

    requests_dir = batch_requests_dir()
    requests_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    next_num = max((b.batch_number for b in state.batches.values()), default=0) + 1
    written_files: list[Path] = []

    for chunk_start in range(0, len(retry_lines), batch_size):
        chunk = retry_lines[chunk_start : chunk_start + batch_size]
        chunk_idx = chunk_start // batch_size + 1
        file_path = requests_dir / f"retry_batch_{chunk_idx:04d}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")

        batch_num = next_num + chunk_idx - 1
        state.batches[file_path.stem] = BatchRecord(
            batch_number=batch_num,
            file_path=str(file_path),
            row_range=f"retry-{len(chunk)}",
            estimated_tokens=len(chunk) * ESTIMATED_TOKENS_PER_REQUEST,
        )
        written_files.append(file_path)
        logger.info("Wrote %s  (%d requests)", file_path.name, len(chunk))

    state.save()


def _cmd_merge(args: argparse.Namespace) -> None:
    from src.context import merged_csv
    from src.logger import setup_logging
    from src.merger import merge_batch_csvs, print_report
    from src.state import PipelineState

    setup_logging()
    state = PipelineState.load()
    output_path = Path(args.output) if args.output else merged_csv()
    merge_batch_csvs(state, output_path)
    print_report(state, output_path)


def _cmd_test(args: argparse.Namespace) -> None:
    """Classify one company synchronously via the Responses API (flex tier when available)."""
    from src.builder import _openai_strict_schema, load_system_prompt, responses_text_format_json_schema
    from src.openai_config import MAX_OUTPUT_TOKENS, PROMPT_CACHE_KEY
    from src.formatter import format_user_message
    from src.logger import setup_logging
    from src.submitter import get_client

    setup_logging()

    data_csv = _resolve_data(args)
    df = pd.read_csv(data_csv)

    if args.company_id:
        match = df[df["org_uuid"] == args.company_id]
    elif args.company_name:
        match = df[df["name"].str.contains(args.company_name, case=False, na=False)]
    else:
        logger.error("Provide --company-id or --company-name.")
        sys.exit(1)

    if match.empty:
        logger.error("No matching company found.")
        sys.exit(1)

    row = match.iloc[0]
    row_dict = row.to_dict()
    user_msg = format_user_message(row_dict, arm=args.arm)

    client = get_client()
    system_prompt = load_system_prompt(args.arm)
    schema = _openai_strict_schema()

    logger.info("Testing classification for: %s (arm=%s)", row_dict.get("name", ""), args.arm)
    print("\n--- USER MESSAGE ---")
    print(user_msg)
    print("--- END USER MESSAGE ---\n")

    import json
    from rich.console import Console
    from rich.panel import Panel
    from tenacity import retry, stop_after_attempt, wait_fixed

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
    def _call(tier: str) -> dict:
        response = client.responses.create(
            model=args.model,
            instructions=system_prompt,
            input=user_msg,
            prompt_cache_key=PROMPT_CACHE_KEY,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
            text=responses_text_format_json_schema(schema),
            service_tier=tier,
        )
        return json.loads(response.output_text)

    try:
        result = _call("flex")
        tier_used = "flex"
    except Exception:
        logger.warning("Flex tier unavailable. Falling back to auto.")
        result = _call("auto")
        tier_used = "auto"

    from src.schema import ClassificationResult
    validated = ClassificationResult.model_validate(result)

    console = Console()
    console.print()
    console.print(Panel(
        "\n".join(f"[bold]{k}[/]: {v}" for k, v in validated.model_dump().items()),
        title=f"Classification Result (arm={args.arm}, service_tier={tier_used})",
        expand=False,
    ))
    console.print()


def _cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline for one arm: prepare, submit, download, merge."""
    from src.logger import setup_logging

    setup_logging()
    args_ns = argparse.Namespace(**vars(args))
    _cmd_prepare(args_ns)

    if args.dry_run:
        return

    _cmd_submit(args_ns)
    _cmd_download(args_ns)
    _cmd_merge(args_ns)


# -- Argument parsing -----------------------------------------------------------


def _parse_rows(rows_str: str | None) -> slice | None:
    if not rows_str:
        return None
    parts = rows_str.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid row range: {rows_str}. Use start:end.")
    return slice(int(parts[0]), int(parts[1]))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    from src.openai_config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL

    parser.add_argument(
        "--arm", required=True, choices=list(VALID_ARMS),
        help="Which experiment arm: 'baseline' (full inputs), 'a' (real name + address), "
             "or 'b' (anonymized name + address).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size",
        help=f"Requests per JSONL file (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to input CSV (default: data/master_csv_directness_experiment.csv)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classify.py",
        description="Directness-experiment classifier: arm-aware OpenAI Batch API runner",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("prepare", help="Build JSONL batch files for the chosen arm")
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print cost breakdown only. No files written, no API calls.")
    p.add_argument("--rows", default=None, help="Row range to process, e.g. '0:50000'")
    p.set_defaults(func=_cmd_prepare)

    p = subs.add_parser("submit", help="Submit pending batches and monitor until complete")
    _add_common_args(p)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max batches in-flight simultaneously (default: 1)")
    p.set_defaults(func=_cmd_submit)

    p = subs.add_parser("status", help="Print status of all tracked batches")
    _add_common_args(p)
    p.set_defaults(func=_cmd_status)

    p = subs.add_parser("download", help="Download results for completed batches")
    _add_common_args(p)
    p.set_defaults(func=_cmd_download)

    p = subs.add_parser("retry", help="Re-submit failed requests as a new batch")
    _add_common_args(p)
    p.set_defaults(func=_cmd_retry)

    p = subs.add_parser("merge", help="Merge batch outputs into the arm's final CSV")
    _add_common_args(p)
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: outputs/<arm>/classified_<arm>.csv)")
    p.set_defaults(func=_cmd_merge)

    p = subs.add_parser("test", help="Classify one company synchronously using flex pricing")
    _add_common_args(p)
    p.add_argument("--company-id", default=None, dest="company_id",
                   help="org_uuid of the company to test")
    p.add_argument("--company-name", default=None, dest="company_name",
                   help="Partial name match (case-insensitive)")
    p.set_defaults(func=_cmd_test)

    p = subs.add_parser("run", help="Full pipeline: prepare -> submit -> download -> merge")
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Run prepare in dry-run mode only")
    p.add_argument("--rows", default=None, help="Row range to process, e.g. '0:50000'")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max batches in-flight simultaneously (default: 1)")
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_active_arm(args.arm)

    try:
        args.func(args)
    except Exception:
        from src.submitter import BillingLimitError
        if sys.exc_info()[0] is BillingLimitError:
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
