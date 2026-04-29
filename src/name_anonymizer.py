"""Deterministic name anonymizer for Arm B (placebo).

Goal: replace the real CompanyName with an opaque, deterministic string that
(a) carries no identity signal the LLM could match against pretraining,
(b) is stable per org_uuid across runs (so Arm B is reproducible), and
(c) cannot leak any token from the original name (Carlini-style memorization
    would let the model recover the original from a partial substring).

The canonical anonymized form is::

    Company-<8-uppercase-hex-from-sha256(org_uuid)>

For example, org_uuid '00007c5c-...' becomes 'Company-A91F3D2E'. This is
the form passed to the LLM as CompanyName under Arm B.
"""

from __future__ import annotations

import hashlib

ANONYMIZED_PREFIX: str = "Company-"
HASH_BYTES: int = 4  # 4 bytes -> 8 hex chars


def anonymize(org_uuid: str) -> str:
    """Return the deterministic anonymized name for the given org_uuid.

    Args:
        org_uuid: The Crunchbase org_uuid string.

    Returns:
        Anonymized company name like 'Company-A91F3D2E'.

    Raises:
        ValueError: If org_uuid is empty or whitespace-only.
    """
    cleaned = (org_uuid or "").strip()
    if not cleaned:
        raise ValueError("Cannot anonymize empty org_uuid")
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return f"{ANONYMIZED_PREFIX}{digest[: HASH_BYTES * 2].upper()}"


def leaks_original_token(anonymized: str, original: str, min_token_len: int = 4) -> bool:
    """Return True if any whitespace-delimited token from *original* appears in *anonymized*.

    Used by tests to assert zero leakage. Tokens shorter than *min_token_len*
    are ignored because incidental hex collisions on 1-3 char strings are
    statistically meaningless.
    """
    haystack = anonymized.lower()
    for token in (original or "").split():
        clean = token.strip(",.()'\"-").lower()
        if len(clean) >= min_token_len and clean in haystack:
            return True
    return False
