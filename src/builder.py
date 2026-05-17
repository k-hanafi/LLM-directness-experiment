"""Build JSONL batch files for the directness experiment.

Each JSONL line is a complete OpenAI **Responses API** request (`POST /v1/responses`)
with an identical prefix (instructions + JSON schema + prompt_cache_key) and one
variable user string produced by the arm-aware formatter.

Three prompt files live in prompts/:
  - baseline_prompt.txt          used by --arm baseline (full inputs)
  - arm_a_prompt.txt             used by --arm a (name + address, training-data recall)
  - arm_b_prompt.txt             used by --arm b (anonymized name, address-based ID)

All three prompt files share their entire body (taxonomy, RAD rules, cohort
rules, analytical scoring criteria, edge cases, and few-shot examples). They
differ only in the INPUT FORMAT block. The test in
tests/test_prompt_consistency.py enforces this invariant.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.openai_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    MAX_BATCH_FILE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_OUTPUT_TOKENS,
    PROMPT_CACHE_KEY,
)
from src.context import batch_requests_dir, project_root
from src.formatter import build_custom_id, format_user_message
from src.schema import ClassificationResult

logger = logging.getLogger(__name__)


def prompt_path_for_arm(arm: str) -> Path:
    """Return the system-prompt file path used by the given arm."""
    mapping = {
        "baseline": "baseline_prompt.txt",
        "a": "arm_a_prompt.txt",
        "b": "arm_b_prompt.txt",
    }
    if arm not in mapping:
        raise ValueError(f"Invalid arm: {arm!r}")
    return project_root() / "prompts" / mapping[arm]


def load_system_prompt(arm: str) -> str:
    """Read the system prompt for the given arm from disk."""
    return prompt_path_for_arm(arm).read_text(encoding="utf-8").strip()


def _openai_strict_schema() -> dict:
    """Generate an OpenAI-compatible JSON schema with strict-mode requirements."""
    schema = ClassificationResult.model_json_schema()
    _add_additional_properties_false(schema)
    return schema


def _add_additional_properties_false(node: dict) -> None:
    """Recursively set additionalProperties: false on all object nodes."""
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
    for value in node.values():
        if isinstance(value, dict):
            _add_additional_properties_false(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _add_additional_properties_false(item)


def responses_text_format_json_schema(schema: dict) -> dict:
    """Responses API structured-output config (`text` parameter body fragment)."""
    return {
        "format": {
            "type": "json_schema",
            "name": "ClassificationResult",
            "strict": True,
            "schema": schema,
        }
    }


def build_request_body(
    user_message: str,
    custom_id: str,
    system_prompt: str,
    schema: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Build one JSONL line for the Batch API (Responses endpoint)."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": system_prompt,
            "input": user_message,
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "text": responses_text_format_json_schema(schema),
        },
    }


def build_batch_files(
    csv_path: str | Path,
    arm: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    row_slice: slice | None = None,
) -> list[Path]:
    """Read the dataset CSV and write JSONL batch files into the active arm's dir.

    Args:
        csv_path: Path to the input CSV.
        arm: 'baseline', 'a', or 'b'. Selects both the prompt and the formatter.
        model: Model name for request bodies.
        batch_size: Requests per JSONL file.
        row_slice: Optional slice to process a subset of rows.

    Returns:
        List of paths to the written JSONL files.
    """
    out_dir = batch_requests_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if row_slice is not None:
        df = df.iloc[row_slice]

    system_prompt = load_system_prompt(arm)
    schema = _openai_strict_schema()
    written_files: list[Path] = []
    max_bytes = min(MAX_FILE_SIZE_MB * 1024 * 1024, MAX_BATCH_FILE_BYTES)

    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        file_path = out_dir / f"batch_{batch_num:04d}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for row_tuple in batch_df.itertuples(index=False):
                row_dict = row_tuple._asdict()
                user_msg = format_user_message(row_dict, arm=arm)
                cid = build_custom_id(str(row_dict.get("org_uuid", "")))
                body = build_request_body(user_msg, cid, system_prompt, schema, model)
                f.write(json.dumps(body, ensure_ascii=False) + "\n")

        file_size = file_path.stat().st_size
        if file_size > max_bytes:
            raise ValueError(
                f"{file_path.name} is {file_size / 1024 / 1024:.1f} MB "
                f"(limit {max_bytes / 1024 / 1024:.1f} MB). "
                "Reduce --batch-size and re-run prepare."
            )

        written_files.append(file_path)
        logger.info("Wrote %s  (%d requests, arm=%s)", file_path.name, len(batch_df), arm)

    return written_files
