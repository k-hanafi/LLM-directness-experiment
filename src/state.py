"""JSON checkpoint for pipeline state, arm-aware.

State files are routed via src.context to outputs/arm_a/state.json or
outputs/arm_b/state.json so the two arms' pipelines never share a
checkpoint and can be driven in any order.

Lifecycle stages per batch:
  prepared -> submitted -> completed | failed | expired

The file is rewritten atomically (write-to-temp then rename) so a crash
mid-write never corrupts the checkpoint.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

from src.context import arm_dir, state_file

logger = logging.getLogger(__name__)

BatchStatus = Literal[
    "prepared",
    "submitted",
    "in_progress",
    "completed",
    "failed",
    "expired",
    "cancelled",
]

_BATCH_RECORD_FIELDS: set[str] = set()


@dataclass
class BatchRecord:
    """One batch's progress through the pipeline."""

    batch_number: int
    file_path: str
    row_range: str
    estimated_tokens: int = 0
    status: BatchStatus = "prepared"
    file_id: str = ""
    batch_id: str = ""
    output_file_id: str = ""
    error_file_id: str = ""
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


_BATCH_RECORD_FIELDS.update(f.name for f in fields(BatchRecord))


@dataclass
class PipelineState:
    """Full pipeline state for one arm, serialised to outputs/arm_X/state.json."""

    run_id: str = ""
    model: str = ""
    arm: str = ""
    total_companies: int = 0
    batches: dict[str, BatchRecord] = field(default_factory=dict)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0

    # -- Persistence -----------------------------------------------------------

    def save(self) -> None:
        """Atomically write state to the active arm's state.json."""
        path = state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls) -> PipelineState:
        """Load the active arm's state from disk, or return a fresh state."""
        path = state_file()
        if not path.exists():
            arm_dir().mkdir(parents=True, exist_ok=True)
            logger.info("No existing state file at %s. Starting fresh.", path)
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))
        state = cls(
            run_id=raw.get("run_id", ""),
            model=raw.get("model", ""),
            arm=raw.get("arm", ""),
            total_companies=raw.get("total_companies", 0),
            total_prompt_tokens=raw.get("total_prompt_tokens", 0),
            total_completion_tokens=raw.get("total_completion_tokens", 0),
            total_cached_tokens=raw.get("total_cached_tokens", 0),
        )
        for key, rec in raw.get("batches", {}).items():
            filtered = {k: v for k, v in rec.items() if k in _BATCH_RECORD_FIELDS}
            state.batches[key] = BatchRecord(**filtered)

        logger.info(
            "Loaded state: run_id=%s, arm=%s, %d batches tracked",
            state.run_id, state.arm, len(state.batches),
        )
        return state

    # -- Convenience queries ---------------------------------------------------

    def pending_batches(self) -> list[BatchRecord]:
        return [b for b in self.batches.values() if b.status == "prepared"]

    def in_flight_batches(self) -> list[BatchRecord]:
        return [
            b for b in self.batches.values()
            if b.status in ("submitted", "in_progress")
        ]

    def completed_batches(self) -> list[BatchRecord]:
        return [b for b in self.batches.values() if b.status == "completed"]

    def failed_batches(self) -> list[BatchRecord]:
        return [
            b for b in self.batches.values()
            if b.status in ("failed", "expired")
        ]

    def estimated_queued_tokens(self) -> int:
        return sum(b.estimated_tokens for b in self.in_flight_batches())
