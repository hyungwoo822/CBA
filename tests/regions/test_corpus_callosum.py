import pytest
from brain_agent.regions.corpus_callosum import CorpusCallosum
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def cc():
    return CorpusCallosum()


# ── Anatomy ──────────────────────────────────────────────────────────

def test_anatomy(cc):
    assert cc.name == "corpus_callosum"
    assert cc.lobe.value == "subcortical"
    assert cc.hemisphere.value == "bilateral"
    assert cc.position.x == 0
    assert cc.position.y == 15
    assert cc.position.z == 5


# ── integrate() ──────────────────────────────────────────────────────

def test_integrate_disjoint_keys(cc):
    left = {"analysis": "plan A", "confidence": 0.8}
    right = {"emotion": "positive", "confidence": 0.6}
    result = cc.integrate(left, right)

    assert result["analysis"] == "plan A"
    assert result["emotion"] == "positive"
    assert result["integration_source"] == "corpus_callosum"
    assert result["left_perspective"] is left
    assert result["right_perspective"] is right


def test_integrate_conflict_left_wins_higher_confidence(cc):
    left = {"assessment": "safe", "confidence": 0.9}
    right = {"assessment": "risky", "confidence": 0.4}
    result = cc.integrate(left, right)

    assert result["assessment"] == "safe"


def test_integrate_conflict_right_wins_higher_confidence(cc):
    left = {"assessment": "safe", "confidence": 0.3}
    right = {"assessment": "risky", "confidence": 0.8}
    result = cc.integrate(left, right)

    assert result["assessment"] == "risky"


def test_integrate_conflict_equal_confidence_left_wins(cc):
    """When confidence is equal, left (analytical) is the tie-breaker."""
    left = {"assessment": "safe", "confidence": 0.5}
    right = {"assessment": "risky", "confidence": 0.5}
    result = cc.integrate(left, right)

    assert result["assessment"] == "safe"


def test_integrate_same_values_no_conflict(cc):
    left = {"tone": "neutral", "confidence": 0.5}
    right = {"tone": "neutral", "confidence": 0.5}
    result = cc.integrate(left, right)

    assert result["tone"] == "neutral"


# ── Transfer counting ───────────────────────────────────────────────

def test_transfer_count_increments(cc):
    assert cc.transfer_count == 0
    cc.integrate({"a": 1}, {"b": 2})
    assert cc.transfer_count == 1
    cc.integrate({"c": 3}, {"d": 4})
    assert cc.transfer_count == 2


# ── async process() ─────────────────────────────────────────────────

async def test_process_integrates_when_both_hemispheres_present(cc):
    sig = Signal(
        type=SignalType.GWT_BROADCAST,
        source="pipeline",
        payload={"text": "hello"},
        metadata={
            "left_result": {"analysis": "greeting", "confidence": 0.7},
            "right_result": {"emotion": "warmth", "confidence": 0.6},
        },
    )
    result = await cc.process(sig)

    assert result is not None
    integrated = result.metadata["integrated_result"]
    assert integrated["analysis"] == "greeting"
    assert integrated["emotion"] == "warmth"
    assert integrated["integration_source"] == "corpus_callosum"
    assert cc.activation_level == pytest.approx(0.8)


async def test_process_passthrough_when_no_hemispheric_data(cc):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="user",
        payload={"text": "test"},
    )
    result = await cc.process(sig)

    assert result is sig
    assert "integrated_result" not in result.metadata
    assert cc.activation_level == pytest.approx(0.1)


async def test_process_passthrough_when_only_left_present(cc):
    sig = Signal(
        type=SignalType.GWT_BROADCAST,
        source="pipeline",
        payload={},
        metadata={"left_result": {"analysis": "data"}},
    )
    result = await cc.process(sig)

    assert "integrated_result" not in result.metadata
    assert cc.activation_level == pytest.approx(0.1)
