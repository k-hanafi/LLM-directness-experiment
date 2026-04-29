"""Tests for Arm A formatter behavior: real CompanyName + real Address only."""

import pytest

from src.formatter import (
    MAX_USER_MESSAGE_CHARS,
    NOT_AVAILABLE,
    _build_address,
    _clean,
    build_custom_id,
    format_user_message,
)


_SAMPLE_ROW = {
    "org_uuid": "abc-123",
    "name": "TestCo",
    "address": "28 Box Street STE N327",
    "city": "New York",
    "state_code": "NY",
    "postal_code": "11222",
    # Fields the directness experiment must NEVER pass to the LLM:
    "short_description": "Should be stripped",
    "description": "Long description should also be stripped",
    "category_list": "AI,Software",
    "category_groups_list": "Technology",
    "founded_date": "01jan2020",
    "year_founded": "2020",
}


class TestArmAStripsDescriptions:
    def test_short_description_is_not_available(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "Short Description: [not available]" in msg
        assert "Should be stripped" not in msg

    def test_long_description_is_not_available(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "Long Description: [not available]" in msg
        assert "Long description should also be stripped" not in msg

    def test_keywords_are_not_available(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "Keywords: [not available]" in msg
        assert "AI,Software" not in msg
        assert "Technology" not in msg

    def test_year_founded_is_not_available(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "YearFounded: [not available]" in msg
        assert "2020" not in msg


class TestArmAPreservesIdentity:
    def test_real_name_passed_through(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "CompanyName: TestCo" in msg

    def test_real_address_concatenated(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "Address: 28 Box Street STE N327, New York, NY, 11222" in msg

    def test_company_id_passed_through(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        assert "CompanyID: abc-123" in msg


class TestAddressBuilder:
    def test_all_components(self):
        row = {"address": "1 Main", "city": "SF", "state_code": "CA", "postal_code": "94102"}
        assert _build_address(row) == "1 Main, SF, CA, 94102"

    def test_missing_address_drops_silently(self):
        row = {"address": "", "city": "SF", "state_code": "CA", "postal_code": "94102"}
        assert _build_address(row) == "SF, CA, 94102"

    def test_missing_postal_drops_silently(self):
        row = {"address": "1 Main", "city": "SF", "state_code": "CA", "postal_code": ""}
        assert _build_address(row) == "1 Main, SF, CA"

    def test_nan_treated_as_missing(self):
        row = {"address": "nan", "city": "SF", "state_code": "CA", "postal_code": "94102"}
        assert _build_address(row) == "SF, CA, 94102"

    def test_all_missing_returns_not_available(self):
        row = {"address": "", "city": "", "state_code": "", "postal_code": ""}
        assert _build_address(row) == NOT_AVAILABLE


class TestFieldOrder:
    """The line order must match the v2 INPUT FORMAT block."""

    def test_arm_a_field_order(self):
        msg = format_user_message(_SAMPLE_ROW, arm="a")
        lines = msg.strip().split("\n")
        assert lines[0].startswith("CompanyID:")
        assert lines[1].startswith("CompanyName:")
        assert lines[2].startswith("Short Description:")
        assert lines[3].startswith("Long Description:")
        assert lines[4].startswith("Address:")
        assert lines[5].startswith("Keywords:")
        assert lines[6].startswith("YearFounded:")


class TestInvalidArm:
    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError):
            format_user_message(_SAMPLE_ROW, arm="c")


class TestTruncation:
    def test_truncation(self):
        row = {**_SAMPLE_ROW, "address": "x" * 20_000}
        msg = format_user_message(row, arm="a")
        assert len(msg) <= MAX_USER_MESSAGE_CHARS + 20
        assert "[truncated]" in msg


class TestBuildCustomId:
    def test_basic(self):
        assert build_custom_id("abc-123") == "directness-abc-123"

    def test_rejects_blank(self):
        with pytest.raises(ValueError):
            build_custom_id("")

    def test_clean_helper(self):
        assert _clean("  hello  ") == "hello"
        assert _clean("nan") == ""
