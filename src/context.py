"""Active-arm context: routes every output path under the right arm directory.

Three arms:
  baseline  -> outputs/baseline/   full input cell (all fields populated)
  a         -> outputs/arm_a/      minimal-input cell, real CompanyName
  b         -> outputs/arm_b/      minimal-input cell, anonymized CompanyName

The three arms run through the SAME pipeline code path under the SAME model
snapshot in this repo; the only experimentally-varied factor is which fields
are populated in the user message. This is the controlled-variables design.

classify.py calls set_active_arm(...) before importing or invoking any module
that reads state. Every path-producing helper reads the active arm at call
time so paths are always arm-correct.
"""

from __future__ import annotations

from pathlib import Path

VALID_ARMS: tuple[str, ...] = ("baseline", "a", "b")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_active_arm: str | None = None


def set_active_arm(arm: str) -> None:
    """Set the active arm for the current process.

    Raises ValueError if arm is not one of VALID_ARMS.
    """
    global _active_arm
    if arm not in VALID_ARMS:
        raise ValueError(f"Invalid arm: {arm!r}. Must be one of {VALID_ARMS}.")
    _active_arm = arm


def active_arm() -> str:
    """Return the active arm. Raises if not yet set."""
    if _active_arm is None:
        raise RuntimeError(
            "Active arm not set. Pass --arm baseline|a|b on the classify.py CLI."
        )
    return _active_arm


def project_root() -> Path:
    return _PROJECT_ROOT


def _arm_dir_name(arm: str) -> str:
    """outputs/<this> for the given arm. Baseline isn't prefixed with 'arm_'."""
    return "baseline" if arm == "baseline" else f"arm_{arm}"


def arm_dir() -> Path:
    """outputs/baseline, outputs/arm_a, or outputs/arm_b for the active arm."""
    return _PROJECT_ROOT / "outputs" / _arm_dir_name(active_arm())


def batch_requests_dir() -> Path:
    return arm_dir() / "batch_requests"


def batch_results_dir() -> Path:
    return arm_dir() / "batch_results"


def batch_errors_dir() -> Path:
    return arm_dir() / "batch_errors"


def batch_outputs_dir() -> Path:
    return arm_dir() / "batch_outputs"


def merged_csv() -> Path:
    """Per-arm final classified CSV.

    outputs/baseline/classified_baseline.csv
    outputs/arm_a/classified_arm_a.csv
    outputs/arm_b/classified_arm_b.csv
    """
    arm = active_arm()
    suffix = "baseline" if arm == "baseline" else f"arm_{arm}"
    return arm_dir() / f"classified_{suffix}.csv"


def state_file() -> Path:
    return arm_dir() / "state.json"


def log_file() -> Path:
    return arm_dir() / "run.log"


def baseline_csv_path() -> Path:
    """Convenience: the baseline run's merged CSV regardless of active arm."""
    return _PROJECT_ROOT / "outputs" / "baseline" / "classified_baseline.csv"


def arm_csv_path(arm: str) -> Path:
    """Convenience: the merged CSV for any arm regardless of active arm."""
    suffix = "baseline" if arm == "baseline" else f"arm_{arm}"
    return _PROJECT_ROOT / "outputs" / _arm_dir_name(arm) / f"classified_{suffix}.csv"


def analysis_dir() -> Path:
    """The shared analysis output directory (not arm-specific)."""
    return _PROJECT_ROOT / "outputs" / "analysis"
