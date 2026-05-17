"""Arm-aware user-message formatter.

Three arms, same field structure in the user message. The Address line is
included in every arm so the only experimentally-varied factor between
baseline and the minimal arms is the populated/[not available] state of
the description / keywords / year fields.

  baseline: real CompanyName, real Short/Long Description, real Address,
            real Keywords, real YearFounded.
  a:        real CompanyName, real Address. Other fields [not available].
  b:        anonymized CompanyName ('Company-<hex>'), real Address.
            Other fields [not available].

The address line concatenates address + city + state_code + postal_code.
Empty components are dropped. The descriptions/keywords/year cells fall
back to '[not available]' if the underlying CSV cell is empty so the
field structure stays uniform across rows. Baseline year may be
``founded_month_year`` (``Mon YYYY`` from the master CSV) or legacy
``year_founded`` / ``founded_date``.
"""

from __future__ import annotations

from typing import Any

from src.name_anonymizer import anonymize

MAX_USER_MESSAGE_CHARS: int = 10_000

NOT_AVAILABLE: str = "[not available]"


def _clean(value: Any) -> str:
    """Convert a value to a stripped string. Treat NaN/None/blank as empty."""
    s = str(value).strip() if value is not None else ""
    if s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def _extract_year(date_str: Any) -> str:
    """Pull a 4-digit year from Crunchbase date strings like '01nov2016'."""
    cleaned = _clean(date_str)
    if not cleaned:
        return ""
    for i in range(len(cleaned) - 3):
        chunk = cleaned[i : i + 4]
        if chunk.isdigit() and 1900 <= int(chunk) <= 2100:
            return chunk
    return cleaned


def _merge_keywords(row: dict[str, Any]) -> str:
    """Combine category_list and category_groups_list into one Keywords field."""
    cats = _clean(row.get("category_list", ""))
    groups = _clean(row.get("category_groups_list", ""))
    if cats and groups:
        return f"{cats}, {groups}"
    return cats or groups


def _build_address(row: dict[str, Any]) -> str:
    """Concatenate address + city + state_code + postal_code into one line.

    Components are separated by ', '. Empty components are dropped.
    Returns NOT_AVAILABLE if every component is empty.
    """
    parts: list[str] = []
    for col in ("address", "city", "state_code", "postal_code"):
        cleaned = _clean(row.get(col, ""))
        if cleaned:
            parts.append(cleaned)
    return ", ".join(parts) if parts else NOT_AVAILABLE


def _short_desc(row: dict[str, Any]) -> str:
    return _clean(row.get("short_description", ""))


def _long_desc(row: dict[str, Any]) -> str:
    """Read long description; prefers master ``long_description``, then Khaled aliases."""
    for key in ("long_description", "Long description", "description"):
        v = _clean(row.get(key, ""))
        if v:
            return v
    return ""


def _year_founded(row: dict[str, Any]) -> str:
    """Value for ``YearFounded:`` line.

    Prefers ``founded_month_year`` from the master CSV (``Mon YYYY``). Otherwise
    uses legacy ``year_founded`` / ``founded_date`` so raw Khaled rows still work.
    """
    canonical = _clean(row.get("founded_month_year", ""))
    if canonical:
        return canonical
    direct = _clean(row.get("year_founded", ""))
    if direct.isdigit() and 1900 <= int(direct) <= 2100:
        return direct
    return _extract_year(row.get("founded_date", ""))


def format_user_message(row: dict[str, Any], arm: str) -> str:
    """Convert one CSV row into the arm-specific user message string.

    Args:
        row: Dictionary whose keys are raw CSV column names.
        arm: 'baseline', 'a', or 'b'.

    Returns:
        A multi-line text block matching the prompt's INPUT FORMAT section,
        with field availability set per arm.
    """
    if arm not in ("baseline", "a", "b"):
        raise ValueError(f"Invalid arm: {arm!r}. Must be 'baseline', 'a', or 'b'.")

    org_uuid = _clean(row.get("org_uuid", ""))
    real_name = _clean(row.get("name", ""))

    if arm == "b":
        company_name = anonymize(org_uuid) if org_uuid else NOT_AVAILABLE
    else:
        company_name = real_name or NOT_AVAILABLE

    address_line = _build_address(row)

    if arm == "baseline":
        short = _short_desc(row) or NOT_AVAILABLE
        long_ = _long_desc(row) or NOT_AVAILABLE
        keywords = _merge_keywords(row) or NOT_AVAILABLE
        year = _year_founded(row) or NOT_AVAILABLE
    else:
        short = NOT_AVAILABLE
        long_ = NOT_AVAILABLE
        keywords = NOT_AVAILABLE
        year = NOT_AVAILABLE

    parts = [
        f"CompanyID: {org_uuid}",
        f"CompanyName: {company_name}",
        f"Short Description: {short}",
        f"Long Description: {long_}",
        f"Address: {address_line}",
        f"Keywords: {keywords}",
        f"YearFounded: {year}",
    ]

    message = "\n".join(parts)

    if len(message) > MAX_USER_MESSAGE_CHARS:
        message = message[:MAX_USER_MESSAGE_CHARS] + "\n[truncated]"

    return message


def build_custom_id(org_uuid: str) -> str:
    """Create a deterministic custom_id for batch result matching.

    The custom_id is the only key joining async batch results back to
    their input row; batch output order is not guaranteed.
    """
    sanitized = _clean(org_uuid).replace(" ", "-")
    if not sanitized:
        raise ValueError("Cannot build custom_id from blank org_uuid")
    return f"directness-{sanitized}"
