"""Structured logging with arm-aware file routing.

Console handler is shared; the rotating file handler writes to
outputs/arm_X/batchfiles/run.log so each arm has its own audit trail.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from src.context import arm_dir, batchfiles_dir, log_file

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with console and arm-scoped file handlers.

    Safe to call multiple times. Only the first invocation attaches handlers.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    arm_dir().mkdir(parents=True, exist_ok=True)
    batchfiles_dir().mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    console.setLevel(level)

    file_handler = RotatingFileHandler(
        log_file(),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s")
    )

    root.addHandler(console)
    root.addHandler(file_handler)

    _CONFIGURED = True
