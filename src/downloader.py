"""Batch result downloader for the active arm.

Result and error files are written under outputs/arm_X/ via src.context so
the two arms' outputs never collide.

Per-response cached_tokens from the usage object are aggregated so the
final report can show actual dollars saved from prompt caching.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.context import batch_errors_dir, batch_outputs_dir, batch_results_dir
from src.schema import ClassificationResult
from src.state import PipelineState
from src.submitter import get_client

logger = logging.getLogger(__name__)


def _download_file(client, file_id: str, dest: Path) -> Path:
    """Download a file from OpenAI and write it to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    dest.write_bytes(content.read())
    logger.info("Downloaded %s -> %s", file_id, dest.name)
    return dest


def _assistant_json_from_batch_body(body: dict) -> str | None:
    """Extract assistant JSON text from a batch line's `response.body`.

    Supports **Responses API** (`output` with `output_text` blocks) and legacy
    **Chat Completions** (`choices[0].message.content`) for older batch files.
    """
    # Responses API — POST /v1/responses
    out_items = body.get("output")
    if out_items is not None:
        parts: list[str] = []
        for item in out_items:
            if item.get("type") != "message":
                continue
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    parts.append(block.get("text") or "")
        text = "".join(parts).strip()
        if text:
            return text

    # Chat Completions — POST /v1/chat/completions (legacy batches)
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

    return None


def _usage_from_batch_body(body: dict) -> dict[str, int]:
    """Normalize per-response usage for cost aggregation (batch discount math)."""
    usage = body.get("usage") or {}
    # Responses API
    if "input_tokens" in usage:
        inp_details = usage.get("input_tokens_details") or {}
        return {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "cached_tokens": int(inp_details.get("cached_tokens") or 0),
        }
    # Chat Completions
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
    }


def _parse_result_line(line: dict) -> dict | None:
    """Extract classification fields and usage stats from one JSONL result line.

    Returns None if the line represents an error response.
    """
    custom_id = line.get("custom_id", "")
    response = line.get("response", {})
    body = response.get("body", {})

    if response.get("status_code") != 200:
        error = line.get("error", response.get("error", {}))
        logger.warning(
            "Non-200 for %s: %s",
            custom_id, error.get("message", "unknown error"),
        )
        return None

    content_str = _assistant_json_from_batch_body(body)
    if not content_str:
        logger.warning("No assistant output in response for %s", custom_id)
        return None

    try:
        parsed = json.loads(content_str)
        ClassificationResult.model_validate(parsed)
    except Exception:
        logger.warning("Validation failed for %s", custom_id, exc_info=True)
        return None

    u = _usage_from_batch_body(body)

    return {
        "custom_id": custom_id,
        "classification": parsed,
        "usage": {
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "cached_tokens": u["cached_tokens"],
        },
    }


def _write_batch_csv(records: list[dict], batch_num: int) -> Path:
    """Write parsed classification results to a per-batch CSV under the active arm."""
    out_dir = batch_outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_{batch_num:04d}.csv"

    if not records:
        logger.warning("No valid records for batch %d. Skipping CSV.", batch_num)
        return path

    fieldnames = list(ClassificationResult.model_fields.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec["classification"])

    logger.info("Wrote %d rows -> %s", len(records), path.name)
    return path


def download_completed(state: PipelineState) -> None:
    """Download results for all completed batches under the active arm."""
    client = get_client()
    completed = state.completed_batches()

    if not completed:
        logger.info("No completed batches to download.")
        return

    results_dir = batch_results_dir()
    errors_dir = batch_errors_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    for rec in completed:
        if rec.error_file_id:
            error_path = errors_dir / f"batch_{rec.batch_number:04d}_errors.jsonl"
            if not error_path.exists():
                _download_file(client, rec.error_file_id, error_path)

        if not rec.output_file_id:
            logger.warning("Batch %d completed but no output_file_id", rec.batch_number)
            continue

        result_path = results_dir / f"batch_{rec.batch_number:04d}.jsonl"

        if result_path.exists():
            logger.info("Batch %d already downloaded. Skipping.", rec.batch_number)
            continue

        _download_file(client, rec.output_file_id, result_path)

        parsed_records: list[dict] = []
        batch_prompt_toks = 0
        batch_completion_toks = 0
        batch_cached_toks = 0

        with open(result_path, encoding="utf-8") as f:
            for line_str in f:
                line = json.loads(line_str.strip())
                result = _parse_result_line(line)
                if result:
                    parsed_records.append(result)
                    batch_prompt_toks += result["usage"]["prompt_tokens"]
                    batch_completion_toks += result["usage"]["completion_tokens"]
                    batch_cached_toks += result["usage"]["cached_tokens"]

        _write_batch_csv(parsed_records, rec.batch_number)

        state.total_prompt_tokens += batch_prompt_toks
        state.total_completion_tokens += batch_completion_toks
        state.total_cached_tokens += batch_cached_toks

        cache_rate = (
            batch_cached_toks / batch_prompt_toks * 100
            if batch_prompt_toks > 0 else 0.0
        )
        logger.info(
            "Batch %d: %d results, %d prompt toks, %d cached (%.1f%% hit rate)",
            rec.batch_number, len(parsed_records),
            batch_prompt_toks, batch_cached_toks, cache_rate,
        )

    state.save()


def collect_failed_custom_ids(state: PipelineState) -> list[str]:
    """Read all error files for the active arm and collect custom_ids that need retry."""
    failed_ids: list[str] = []
    errors_dir = batch_errors_dir()

    for rec in state.completed_batches() + state.failed_batches():
        error_path = errors_dir / f"batch_{rec.batch_number:04d}_errors.jsonl"
        if not error_path.exists():
            continue
        with open(error_path, encoding="utf-8") as f:
            for line_str in f:
                line = json.loads(line_str.strip())
                cid = line.get("custom_id", "")
                if cid:
                    failed_ids.append(cid)

    logger.info("Collected %d failed custom_ids for retry", len(failed_ids))
    return failed_ids
