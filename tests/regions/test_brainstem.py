import pytest
from brain_agent.regions.brainstem import Brainstem
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def brainstem():
    return Brainstem()


def test_anatomy(brainstem):
    assert brainstem.lobe == Lobe.BRAINSTEM
    assert brainstem.hemisphere == Hemisphere.BILATERAL
    assert brainstem.position.x == 0
    assert brainstem.position.y == -30
    assert brainstem.position.z == -25


def test_initial_state(brainstem):
    assert brainstem.arousal_state == "awake"


async def test_external_input_wakes_up(brainstem):
    brainstem.arousal_state = "sleep"
    brainstem._idle_count = 15
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    result = await brainstem.process(sig)
    assert brainstem.arousal_state == "awake"
    assert brainstem._idle_count == 0
    assert result.metadata["arousal_state"] == "awake"
    assert brainstem.activation_level == pytest.approx(0.8)


async def test_idle_transitions_to_drowsy(brainstem):
    for _ in range(6):
        sig = Signal(
            type=SignalType.RESOURCE_STATUS,
            source="hypothalamus",
            payload={},
        )
        await brainstem.process(sig)
    assert brainstem.arousal_state == "drowsy"
    assert brainstem.activation_level == pytest.approx(0.3)


async def test_idle_transitions_to_sleep(brainstem):
    for _ in range(11):
        sig = Signal(
            type=SignalType.RESOURCE_STATUS,
            source="hypothalamus",
            payload={},
        )
        await brainstem.process(sig)
    assert brainstem.arousal_state == "sleep"
    assert brainstem.activation_level == pytest.approx(0.1)


async def test_wake_resets_idle(brainstem):
    # First go drowsy
    for _ in range(6):
        sig = Signal(
            type=SignalType.RESOURCE_STATUS,
            source="hypothalamus",
            payload={},
        )
        await brainstem.process(sig)
    assert brainstem.arousal_state == "drowsy"

    # Wake up
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "wake up"},
    )
    await brainstem.process(sig)
    assert brainstem.arousal_state == "awake"
    assert brainstem._idle_count == 0


async def test_other_signal_passthrough(brainstem):
    sig = Signal(
        type=SignalType.PLAN,
        source="pfc",
        payload={"goal": "test"},
    )
    result = await brainstem.process(sig)
    assert result is sig
    assert result.metadata["arousal_state"] == "awake"


async def test_resource_status_early_stays_awake(brainstem):
    for _ in range(3):
        sig = Signal(
            type=SignalType.RESOURCE_STATUS,
            source="hypothalamus",
            payload={},
        )
        await brainstem.process(sig)
    assert brainstem.arousal_state == "awake"
    assert brainstem.activation_level == pytest.approx(0.5)
