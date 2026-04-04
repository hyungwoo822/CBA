import pytest
from brain_agent.regions.broca import BrocaArea
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def broca():
    return BrocaArea()


def test_anatomy(broca):
    assert broca.lobe == Lobe.FRONTAL
    assert broca.hemisphere == Hemisphere.LEFT
    assert broca.position.x == -30
    assert broca.position.y == 40
    assert broca.position.z == 15


async def test_formats_plan_response(broca):
    sig = Signal(
        type=SignalType.PLAN,
        source="prefrontal_cortex",
        payload={
            "goal": "test",
            "actions": [{"tool": "respond", "args": {"text": "  hello   world  "}}],
        },
    )
    result = await broca.process(sig)
    assert result.payload["actions"][0]["args"]["text"] == "hello world"


async def test_formats_action_selected(broca):
    sig = Signal(
        type=SignalType.ACTION_SELECTED,
        source="basal_ganglia",
        payload={
            "actions": [{"tool": "respond", "args": {"text": "\n\n\nhello\n\n\n\nworld\n\n\n"}}],
        },
    )
    result = await broca.process(sig)
    text = result.payload["actions"][0]["args"]["text"]
    assert text == "hello\n\nworld"


async def test_formats_response_text_in_payload(broca):
    sig = Signal(
        type=SignalType.PLAN,
        source="prefrontal_cortex",
        payload={"response_text": "   spaced   out   "},
    )
    result = await broca.process(sig)
    assert result.payload["response_text"] == "spaced out"


async def test_passthrough_external_input(broca):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "  keep spaces  "},
    )
    result = await broca.process(sig)
    assert result is sig
    assert result.payload["text"] == "  keep spaces  "


async def test_emits_activation(broca):
    sig = Signal(
        type=SignalType.PLAN,
        source="prefrontal_cortex",
        payload={"actions": [{"tool": "respond", "args": {"text": "hi"}}]},
    )
    await broca.process(sig)
    assert broca.activation_level == pytest.approx(0.6)


async def test_no_actions_key(broca):
    sig = Signal(
        type=SignalType.PLAN,
        source="prefrontal_cortex",
        payload={"goal": "test"},
    )
    result = await broca.process(sig)
    assert result is not None


async def test_action_without_text_arg(broca):
    sig = Signal(
        type=SignalType.PLAN,
        source="prefrontal_cortex",
        payload={"actions": [{"tool": "shell", "args": {"command": "ls"}}]},
    )
    result = await broca.process(sig)
    assert result.payload["actions"][0]["args"]["command"] == "ls"
