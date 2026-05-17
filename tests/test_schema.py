"""Tests for ClassificationResult schema enforcement.

Identical to the parent v2 tests since the output schema must be byte-identical
for the directness experiment outputs to merge against the v2 baseline CSV.
"""

import pytest
from pydantic import ValidationError

from src.schema import ClassificationResult

_VALID = {
    "CompanyID": "test-001",
    "CompanyName": "Acme AI",
    "ai_native": 1,
    "subclass": "1B",
    "rad_score": "RAD-M",
    "cohort": "PRE-GENAI",
    "conf_classification": 4,
    "conf_rad": 3,
    "reasons_3_points": "Point A | Point B | Point C",
    "sources_used": "name, address",
    "verification_critique": "Borderline 1B vs 1D.",
    "pretraining_inferred_description": None,
}


def _make(**overrides) -> dict:
    return {**_VALID, **overrides}


class TestValidInput:
    def test_accepts_valid_ai_native(self):
        result = ClassificationResult.model_validate(_make())
        assert result.ai_native == 1

    def test_accepts_all_subclasses(self):
        for sc in ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C", "0"]:
            ClassificationResult.model_validate(_make(subclass=sc))

    def test_accepts_all_rad_scores(self):
        for rad in ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]:
            ClassificationResult.model_validate(_make(rad_score=rad))

    def test_conf_rad_null_for_rad_na(self):
        result = ClassificationResult.model_validate(
            _make(ai_native=0, subclass="0A", rad_score="RAD-NA", conf_rad=None)
        )
        assert result.conf_rad is None

    def test_pretraining_description_accepts_string(self):
        result = ClassificationResult.model_validate(
            _make(pretraining_inferred_description="AI infrastructure company in SF.")
        )
        assert result.pretraining_inferred_description == "AI infrastructure company in SF."

    def test_pretraining_description_accepts_null(self):
        result = ClassificationResult.model_validate(
            _make(pretraining_inferred_description=None)
        )
        assert result.pretraining_inferred_description is None


class TestInvalidInput:
    def test_rejects_invalid_subclass(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(subclass="2A"))

    def test_rejects_invalid_rad_score(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(rad_score="RAD-X"))

    def test_rejects_confidence_zero(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(conf_classification=0))

    def test_rejects_confidence_six(self):
        with pytest.raises(ValidationError):
            ClassificationResult.model_validate(_make(conf_rad=6))


class TestSchemaGeneration:
    def test_schema_has_all_fields(self):
        schema = ClassificationResult.model_json_schema()
        props = schema["properties"]
        expected = [
            "CompanyID", "CompanyName", "ai_native", "subclass", "rad_score",
            "cohort", "conf_classification", "conf_rad", "reasons_3_points",
            "sources_used", "verification_critique",
            "pretraining_inferred_description",
        ]
        for field_name in expected:
            assert field_name in props

    def test_schema_field_count(self):
        schema = ClassificationResult.model_json_schema()
        assert len(schema["properties"]) == 12
