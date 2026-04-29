"""Tests for the deterministic name anonymizer used by Arm B."""

import re

import pytest

from src.name_anonymizer import (
    ANONYMIZED_PREFIX,
    HASH_BYTES,
    anonymize,
    leaks_original_token,
)


class TestDeterminism:
    def test_same_uuid_same_output(self):
        assert anonymize("abc-123") == anonymize("abc-123")

    def test_different_uuid_different_output(self):
        assert anonymize("abc-123") != anonymize("abc-124")

    def test_output_format(self):
        out = anonymize("00007c5c-9260-0dfb-c160-89a416f1a7cc")
        pattern = re.compile(rf"^{re.escape(ANONYMIZED_PREFIX)}[0-9A-F]{{{HASH_BYTES * 2}}}$")
        assert pattern.match(out), f"Unexpected format: {out!r}"


class TestEmptyInput:
    def test_empty_uuid_raises(self):
        with pytest.raises(ValueError):
            anonymize("")

    def test_whitespace_uuid_raises(self):
        with pytest.raises(ValueError):
            anonymize("   ")


class TestNoLeakage:
    """Every anonymized name must be free of any meaningful token from the original."""

    def test_anonymized_drops_original_long_token(self):
        original = "OpenAI Cognition Labs"
        out = anonymize("ex-001")
        assert not leaks_original_token(out, original, min_token_len=4)

    def test_anonymized_drops_punctuation_variants(self):
        original = "Resilio (acquired by meQuilibrium)"
        out = anonymize("ex-002")
        assert not leaks_original_token(out, original, min_token_len=4)

    def test_short_tokens_ignored(self):
        out = anonymize("ex-003")
        assert not leaks_original_token(out, "AI ML LLM", min_token_len=4)


class TestProductionScale:
    """Sanity-check the namespace size is large enough to avoid collisions on 269k rows."""

    def test_no_collisions_on_thousand_uuids(self):
        seen = set()
        for i in range(1000):
            out = anonymize(f"uuid-{i:06d}")
            assert out not in seen, f"Collision on iteration {i}: {out}"
            seen.add(out)
