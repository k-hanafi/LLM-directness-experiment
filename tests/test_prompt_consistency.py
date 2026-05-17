"""Enforce the experimental control: all three prompts must be byte-identical
EXCEPT for the INPUT FORMAT block.

If anyone tweaks taxonomy definitions, RAD rules, cohort rules, edge cases,
or few-shot examples in only one of the three prompts, this test will fail
loudly. That keeps the only experimentally-varied factor cleanly localised
to the input-format description.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
BASELINE = PROMPTS / "baseline_prompt.txt"
ARM_A = PROMPTS / "arm_a_prompt.txt"
ARM_B = PROMPTS / "arm_b_prompt.txt"

INPUT_FORMAT_RE = re.compile(
    r"═+\s*\nINPUT FORMAT\s*\n═+\s*\n.*?(?=\n═+\s*\nDIMENSION 1:)",
    re.DOTALL,
)

ALL_PROMPTS = [
    ("baseline", BASELINE),
    ("arm_a", ARM_A),
    ("arm_b", ARM_B),
]


def _strip_input_format_block(text: str) -> str:
    """Replace the INPUT FORMAT block with a placeholder, leaving the rest intact."""
    return INPUT_FORMAT_RE.sub("<<INPUT_FORMAT_BLOCK>>\n", text, count=1)


class TestPromptInvariance:
    def test_all_prompts_exist(self):
        for name, path in ALL_PROMPTS:
            assert path.exists(), f"missing {name} prompt at {path}"

    @pytest.mark.parametrize("name_a,path_a,name_b,path_b", [
        ("baseline", BASELINE, "arm_a", ARM_A),
        ("baseline", BASELINE, "arm_b", ARM_B),
        ("arm_a", ARM_A, "arm_b", ARM_B),
    ])
    def test_input_format_block_is_distinct(self, name_a, path_a, name_b, path_b):
        a_text = path_a.read_text(encoding="utf-8")
        b_text = path_b.read_text(encoding="utf-8")
        a_block = INPUT_FORMAT_RE.search(a_text)
        b_block = INPUT_FORMAT_RE.search(b_text)
        assert a_block is not None, f"INPUT FORMAT block not found in {name_a}"
        assert b_block is not None, f"INPUT FORMAT block not found in {name_b}"
        assert a_block.group() != b_block.group(), (
            f"The INPUT FORMAT blocks of {name_a} and {name_b} should differ; "
            "otherwise the experiment is not testing different input conditions."
        )

    @pytest.mark.parametrize("name,path", [
        ("arm_a", ARM_A),
        ("arm_b", ARM_B),
    ])
    def test_everything_outside_input_format_matches_baseline(self, name, path):
        b_stripped = _strip_input_format_block(BASELINE.read_text(encoding="utf-8"))
        m_stripped = _strip_input_format_block(path.read_text(encoding="utf-8"))
        if b_stripped != m_stripped:
            import difflib
            diff = "\n".join(list(difflib.unified_diff(
                b_stripped.splitlines(),
                m_stripped.splitlines(),
                fromfile="baseline_prompt.txt",
                tofile=path.name,
                lineterm="",
            ))[:80])
            raise AssertionError(
                f"baseline_prompt and {name} differ outside the "
                "INPUT FORMAT block. The only experimentally-varied factor must "
                "be the input-format description.\n\nFirst lines of diff:\n" + diff,
            )

    @pytest.mark.parametrize("name,path", [
        ("arm_a", ARM_A),
        ("arm_b", ARM_B),
    ])
    def test_minimal_prompts_mark_fields_not_available(self, name, path):
        text = path.read_text(encoding="utf-8")
        block = INPUT_FORMAT_RE.search(text).group()
        assert "[not available]" in block, (
            f"{name} prompt's INPUT FORMAT block should explicitly mark "
            "the stripped fields as [not available]."
        )

    def test_baseline_prompt_does_not_mark_fields_not_available_in_input_format(self):
        b = BASELINE.read_text(encoding="utf-8")
        block = INPUT_FORMAT_RE.search(b).group()
        assert "[not available]" not in block.replace(
            'unless explicitly marked "[not available]"', ""
        ), (
            "Baseline prompt's INPUT FORMAT block should not mark fields as "
            "[not available] because the baseline cell sees full inputs."
        )

    def test_arm_a_encourages_training_data_recall(self):
        text = ARM_A.read_text(encoding="utf-8")
        block = INPUT_FORMAT_RE.search(text).group()
        assert "training" in block.lower(), (
            "Arm A prompt should encourage the model to use training data "
            "knowledge about the company name."
        )

    def test_arm_b_explains_anonymized_names(self):
        text = ARM_B.read_text(encoding="utf-8")
        block = INPUT_FORMAT_RE.search(text).group()
        assert "anonymized" in block.lower() or "anonymised" in block.lower(), (
            "Arm B prompt should explain that company names are anonymized."
        )
        assert "address" in block.lower(), (
            "Arm B prompt should instruct the model to use the address "
            "to identify the company."
        )
