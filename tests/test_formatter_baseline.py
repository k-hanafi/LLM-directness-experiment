"""Tests for the baseline arm formatter behavior: full inputs preserved."""

import pytest

from src.formatter import (
    NOT_AVAILABLE,
    format_user_message,
)


_SAMPLE_ROW = {
    "org_uuid": "abc-123",
    "name": "TestCo",
    "address": "28 Box Street STE N327",
    "city": "New York",
    "state_code": "NY",
    "postal_code": "11222",
    "short_description": "An AI-native test company",
    "description": "Long description for the test company about AI things",
    "category_list": "AI,Software",
    "category_groups_list": "Technology",
    "founded_date": "01jan2020",
}


class TestBaselineFullInputs:
    def test_short_description_preserved(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "Short Description: An AI-native test company" in msg

    def test_long_description_preserved(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "Long Description: Long description for the test company about AI things" in msg

    def test_keywords_merged(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "Keywords: AI,Software, Technology" in msg

    def test_year_extracted(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "YearFounded: 2020" in msg

    def test_address_concatenated(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "Address: 28 Box Street STE N327, New York, NY, 11222" in msg

    def test_real_name_passed_through(self):
        msg = format_user_message(_SAMPLE_ROW, arm="baseline")
        assert "CompanyName: TestCo" in msg


class TestBaselineMissingFields:
    def test_missing_long_description_falls_back(self):
        row = {**_SAMPLE_ROW, "description": ""}
        msg = format_user_message(row, arm="baseline")
        assert f"Long Description: {NOT_AVAILABLE}" in msg

    def test_missing_keywords_falls_back(self):
        row = {**_SAMPLE_ROW, "category_list": "", "category_groups_list": ""}
        msg = format_user_message(row, arm="baseline")
        assert f"Keywords: {NOT_AVAILABLE}" in msg

    def test_missing_year_falls_back(self):
        row = {**_SAMPLE_ROW, "founded_date": "nan"}
        row.pop("year_founded", None)
        msg = format_user_message(row, arm="baseline")
        assert f"YearFounded: {NOT_AVAILABLE}" in msg

    def test_long_description_alt_column_name(self):
        row = {**_SAMPLE_ROW}
        row.pop("description")
        row["Long description"] = "Alternative long description column"
        msg = format_user_message(row, arm="baseline")
        assert "Long Description: Alternative long description column" in msg

    def test_long_description_master_column(self):
        row = {**_SAMPLE_ROW}
        row.pop("description")
        row["long_description"] = "Coalesced long text from master CSV"
        msg = format_user_message(row, arm="baseline")
        assert "Long Description: Coalesced long text from master CSV" in msg

    def test_long_description_prefers_master_column_over_description(self):
        row = {**_SAMPLE_ROW, "long_description": "Master wins"}
        msg = format_user_message(row, arm="baseline")
        assert "Long Description: Master wins" in msg

    def test_founded_month_year_from_master_csv(self):
        row = {**_SAMPLE_ROW}
        row.pop("founded_date", None)
        row.pop("year_founded", None)
        row["founded_month_year"] = "Nov 2016"
        msg = format_user_message(row, arm="baseline")
        assert "YearFounded: Nov 2016" in msg


class TestMasterCsvColumnSet:
    """Rows shaped like ``master_csv_directness_experiment.csv`` (no legacy founding keys)."""

    def test_baseline_full_message_with_master_columns_only(self):
        row = {
            "org_uuid": "org-1",
            "name": "Acme",
            "address": "1 Rd",
            "city": "Austin",
            "state_code": "TX",
            "postal_code": "78701",
            "short_description": "Widgets",
            "long_description": "A long narrative.",
            "category_list": "Hardware",
            "category_groups_list": "Manufacturing",
            "founded_month_year": "Mar 2021",
        }
        msg = format_user_message(row, arm="baseline")
        assert "YearFounded: Mar 2021" in msg
        assert "Long Description: A long narrative." in msg
        assert "Keywords: Hardware, Manufacturing" in msg


class TestFieldOrderConsistency:
    """All three arms must emit the same line ordering so the prompt's
    INPUT FORMAT block matches the user message structure exactly."""

    def test_all_arms_same_field_order(self):
        for arm in ("baseline", "a", "b"):
            msg = format_user_message(_SAMPLE_ROW, arm=arm)
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
