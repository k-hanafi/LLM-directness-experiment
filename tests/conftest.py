"""Shared pytest fixtures: every test file exercises a known active arm.

Tests that need a different arm can call set_active_arm(...) themselves;
this autouse fixture only sets a default to keep context functions working
in tests that don't care about the specific arm.
"""

import pytest

from src.context import set_active_arm


@pytest.fixture(autouse=True)
def _default_arm():
    """Set arm='baseline' before each test so context functions don't raise."""
    set_active_arm("baseline")
    yield
