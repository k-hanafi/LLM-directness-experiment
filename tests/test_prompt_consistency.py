"""Enforce the experimental control: baseline_prompt and arms_a_b_prompt
must be byte-identical EXCEPT for the INPUT FORMAT block.

If anyone tweaks taxonomy definitions, RAD rules, cohort rules, edge cases,
or few-shot examples in only one of the two prompts, this test will fail
loudly. That keeps the only experimentally-varied factor cleanly localised
to the input-format description.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
BASELINE = PROMPTS / "baseline_prompt.txt"
ARMS_AB = PROMPTS / "arms_a_b_prompt.txt"

INPUT_FORMAT_RE = re.compile(
    r"═+\s*\nINPUT FORMAT\s*\n═+\s*\n.*?(?=\n═+\s*\nDIMENSION 1:)",
    re.DOTALL,
)


def _strip_input_format_block(text: str) -> str:
    """Replace the INPUT FORMAT block with a placeholder, leaving the rest intact."""
    return INPUT_FORMAT_RE.sub("<<INPUT_FORMAT_BLOCK>>\n", text, count=1)


class TestPromptInvariance:
    def test_both_prompts_exist(self):
        assert BASELINE.exists(), f"missing {BASELINE}"
        assert ARMS_AB.exists(), f"missing {ARMS_AB}"

    def test_input_format_block_is_distinct(self):
        b = BASELINE.read_text(encoding="utf-8")
        m = ARMS_AB.read_text(encoding="utf-8")
        b_block = INPUT_FORMAT_RE.search(b)
        m_block = INPUT_FORMAT_RE.search(m)
        assert b_block is not None, "INPUT FORMAT block not found in baseline prompt"
        assert m_block is not None, "INPUT FORMAT block not found in arms_a_b prompt"
        assert b_block.group() != m_block.group(), (
            "The INPUT FORMAT blocks should differ; otherwise the two prompts "
            "are identical and the experiment is not testing input availability."
        )

    def test_everything_outside_input_format_is_byte_identical(self):
        b_stripped = _strip_input_format_block(BASELINE.read_text(encoding="utf-8"))
        m_stripped = _strip_input_format_block(ARMS_AB.read_text(encoding="utf-8"))
        if b_stripped != m_stripped:
            import difflib
            diff = "\n".join(difflib.unified_diff(
                b_stripped.splitlines(),
                m_stripped.splitlines(),
                fromfile="baseline_prompt.txt",
                tofile="arms_a_b_prompt.txt",
                lineterm="",
            )[:80])
            raise AssertionError(
                "baseline_prompt and arms_a_b_prompt differ outside the "
                "INPUT FORMAT block. The only experimentally-varied factor must "
                "be the input-format description.\n\nFirst lines of diff:\n" + diff,
            )

    def test_arms_a_b_prompt_marks_fields_not_available(self):
        m = ARMS_AB.read_text(encoding="utf-8")
        block = INPUT_FORMAT_RE.search(m).group()
        assert "[not available]" in block, (
            "arms_a_b_prompt's INPUT FORMAT block should explicitly mark "
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
