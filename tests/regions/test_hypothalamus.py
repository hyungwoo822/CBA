import pytest
from brain_agent.regions.hypothalamus import Hypothalamus
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.neuromodulators import Neuromodulators


@pytest.fixture
def neuromod():
    return Neuromodulators()


@pytest.fixture
def hypothalamus(neuromod):
    return Hypothalamus(neuromodulators=neuromod)


async def test_no_direct_neuromodulator_update(hypothalamus, neuromod):
    """Hypothalamus no longer directly updates neuromodulators.
    NeuromodulatorController in the pipeline handles this now."""
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 5, "staging_count": 10, "error_rate": 0.2})
    await hypothalamus.process(sig)
    # Neuromodulators should remain at default values (controller handles updates)
    assert neuromod.urgency == pytest.approx(0.5)
    assert neuromod.patience == pytest.approx(0.5)


async def test_tracks_pending_requests(hypothalamus):
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 7, "staging_count": 5, "error_rate": 0.1})
    await hypothalamus.process(sig)
    assert hypothalamus.pending_requests == 7
    assert hypothalamus.staging_count == 5
    assert hypothalamus.error_rate == pytest.approx(0.1)


async def test_triggers_consolidation_on_high_staging(hypothalamus):
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 0, "staging_count": 25, "error_rate": 0.0})
    result = await hypothalamus.process(sig)
    assert result is not None
    assert result.type == SignalType.CONSOLIDATION_TRIGGER
    assert result.payload["staging_count"] == 25


async def test_no_consolidation_when_staging_low(hypothalamus):
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 0, "staging_count": 10, "error_rate": 0.0})
    result = await hypothalamus.process(sig)
    assert result is None


async def test_returns_none_for_non_resource_signal(hypothalamus):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user",
                 payload={"text": "hello"})
    result = await hypothalamus.process(sig)
    assert result is None


async def test_state_tracked_without_neuromod_update(hypothalamus, neuromod):
    """Hypothalamus tracks state but does not update neuromodulators directly."""
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 20, "staging_count": 0, "error_rate": 0.0})
    await hypothalamus.process(sig)
    assert hypothalamus.pending_requests == 20
    # Neuromodulators unchanged (controller handles updates)
    assert neuromod.urgency == pytest.approx(0.5)


async def test_emits_activation(hypothalamus):
    sig = Signal(type=SignalType.RESOURCE_STATUS, source="monitor",
                 payload={"pending_requests": 0, "staging_count": 0, "error_rate": 0.0})
    await hypothalamus.process(sig)
    assert hypothalamus.activation_level == pytest.approx(0.3)


async def test_name_and_position(hypothalamus):
    assert hypothalamus.name == "hypothalamus"
    assert hypothalamus.position.y == -10
    assert hypothalamus.position.z == -15
