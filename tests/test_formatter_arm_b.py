"""Tests for Arm B (placebo) formatter behavior: anonymized CompanyName + real Address."""

from src.formatter import format_user_message
from src.name_anonymizer import anonymize, leaks_original_token


_SAMPLE_ROW = {
    "org_uuid": "abc-123",
    "name": "OpenAI",
    "address": "3180 18th Street",
    "city": "San Francisco",
    "state_code": "CA",
    "postal_code": "94110",
    "short_description": "Should be stripped",
    "description": "Long description",
    "category_list": "AI,ML",
    "category_groups_list": "Software",
    "founded_date": "01jan2015",
}


class TestArmBAnonymizesName:
    def test_company_name_is_anonymized(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        expected = anonymize("abc-123")
        assert f"CompanyName: {expected}" in msg

    def test_real_name_does_not_appear(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        assert "OpenAI" not in msg

    def test_anonymized_name_carries_no_original_token(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        anon_name = anonymize("abc-123")
        assert not leaks_original_token(anon_name, _SAMPLE_ROW["name"])


class TestArmBPreservesAddress:
    def test_real_address_passed_through(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        assert "Address: 3180 18th Street, San Francisco, CA, 94110" in msg

    def test_company_id_passed_through(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        assert "CompanyID: abc-123" in msg


class TestArmBStripsDescriptions:
    def test_descriptions_stripped(self):
        msg = format_user_message(_SAMPLE_ROW, arm="b")
        assert "Short Description: [not available]" in msg
        assert "Long Description: [not available]" in msg
        assert "Keywords: [not available]" in msg
        assert "YearFounded: [not available]" in msg
        assert "Should be stripped" not in msg


class TestArmAVsArmBDifference:
    """Confirm Arm A and Arm B differ ONLY in the CompanyName line."""

    def test_only_company_name_differs(self):
        msg_a = format_user_message(_SAMPLE_ROW, arm="a")
        msg_b = format_user_message(_SAMPLE_ROW, arm="b")
        lines_a = msg_a.split("\n")
        lines_b = msg_b.split("\n")
        assert len(lines_a) == len(lines_b)
        for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
            if i == 1:
                assert la != lb, "CompanyName line must differ between arms"
                assert la.startswith("CompanyName: ")
                assert lb.startswith("CompanyName: ")
            else:
                assert la == lb, f"Line {i} unexpectedly differs: A={la!r}  B={lb!r}"
