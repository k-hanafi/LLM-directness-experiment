"""Pipeline configuration.

Output paths are arm-aware and routed via src.context, not hardcoded constants.
The system prompt cache key is shared across arms because both prompts share
their long body (taxonomy + few-shot); only the short INPUT FORMAT block
differs, which makes prefix caching effective even across arms.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[1] / "keys" / "openai.env"
load_dotenv(_ENV_FILE)

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "gpt-5.4-nano"

PROMPT_CACHE_KEY: str = "directness-experiment-system-prompt"

# ---------------------------------------------------------------------------
# Tier 5 rate limits
# ---------------------------------------------------------------------------

MAX_REQUESTS_PER_MINUTE: int = 30_000
MAX_TOKENS_PER_MINUTE: int = 180_000_000

MAX_BATCH_QUEUE_TOKENS: int = 15_000_000_000
MAX_REQUESTS_PER_BATCH: int = 50_000
BATCH_CREATION_PER_HOUR: int = 2_000
MAX_FILE_SIZE_MB: int = 190

# ---------------------------------------------------------------------------
# Per-request budget
# ---------------------------------------------------------------------------

ESTIMATED_TOKENS_PER_REQUEST: int = 7_500

MAX_OUTPUT_TOKENS: int = 450

# ---------------------------------------------------------------------------
# Batch construction defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 5_000
