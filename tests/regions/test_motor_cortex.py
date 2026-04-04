"""Tests for Primary Motor Cortex (M1) — final output execution."""
import pytest
from brain_agent.regions.motor_cortex import MotorCortex
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def m1():
    return MotorCortex()


def test_anatomy(m1):
    assert m1.name == "motor_cortex"
    assert m1.lobe == Lobe.FRONTAL
    assert m1.hemisphere == Hemisphere.LEFT


async def test_cleans_whitespace(m1):
    """M1 should normalize whitespace in output."""
    sig = Signal(
        type=SignalType.PLAN,
        source="broca",
        payload={"actions": [{"tool": "respond", "args": {"text": "  hello   world  "}}]},
    )
    result = await m1.process(sig)
    assert result.payload["actions"][0]["args"]["text"] == "hello world"


async def test_collapses_newlines(m1):
    """M1 should collapse excessive newlines."""
    sig = Signal(
        type=SignalType.PLAN,
        source="broca",
        payload={"actions": [{"tool": "respond", "args": {"text": "line1\n\n\n\nline2"}}]},
    )
    result = await m1.process(sig)
    assert result.payload["actions"][0]["args"]["text"] == "line1\n\nline2"


async def test_removes_control_chars(m1):
    """M1 should strip control characters."""
    sig = Signal(
        type=SignalType.PLAN,
        source="broca",
        payload={"actions": [{"tool": "respond", "args": {"text": "hello\x00world"}}]},
    )
    result = await m1.process(sig)
    assert "\x00" not in result.payload["actions"][0]["args"]["text"]


async def test_ignores_non_plan_signals(m1):
    """M1 should pass through non-PLAN/ACTION_SELECTED signals unchanged."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "hello"},
    )
    result = await m1.process(sig)
    assert result.payload["text"] == "hello"


async def test_formats_response_text(m1):
    """M1 should also format top-level response_text."""
    sig = Signal(
        type=SignalType.PLAN,
        source="broca",
        payload={"response_text": "  multiple   spaces  "},
    )
    result = await m1.process(sig)
    assert result.payload["response_text"] == "multiple spaces"
