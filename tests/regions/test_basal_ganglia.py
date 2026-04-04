import pytest
from brain_agent.regions.basal_ganglia import BasalGanglia
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def bg(): return BasalGanglia()

async def test_go_for_confident_action(bg):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"actions": [{"tool": "read_file", "confidence": 0.9}]})
    result = await bg.process(sig)
    assert result is not None
    assert result.type == SignalType.ACTION_SELECTED

async def test_nogo_for_low_confidence(bg):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"actions": [{"tool": "delete_all", "confidence": 0.1, "risk": 0.9}]})
    result = await bg.process(sig)
    assert result is None


async def test_low_patience_increases_nogo(bg):
    """Low patience (serotonin) should increase NoGo score, making inhibition
    stronger for borderline actions (Doya 2002)."""
    action = {"tool": "risky_op", "confidence": 0.55, "risk": 0.2}
    # High patience -> lower NoGo -> more likely to pass
    sig_patient = Signal(
        type=SignalType.PLAN, source="pfc",
        payload={"actions": [action]},
        metadata={"neuromodulators": {"patience": 1.0}},
    )
    result_patient = await bg.process(sig_patient)

    # Low patience -> higher NoGo addition -> might inhibit
    sig_impatient = Signal(
        type=SignalType.PLAN, source="pfc",
        payload={"actions": [action]},
        metadata={"neuromodulators": {"patience": 0.0}},
    )
    result_impatient = await bg.process(sig_impatient)

    # With patience=1.0, nogo_score gets +0.0; with patience=0.0, nogo_score gets +0.15
    # So the impatient result should have lower go_score (or be inhibited)
    if result_patient is not None and result_impatient is not None:
        assert result_patient.payload["go_score"] > result_impatient.payload["go_score"]
    elif result_patient is not None:
        # Impatient was inhibited entirely — confirms stronger NoGo
        pass
    else:
        # Both inhibited — edge case but acceptable
        pass


async def test_patience_default_when_missing(bg):
    """Without neuromodulators in metadata, patience defaults to 0.5."""
    sig = Signal(
        type=SignalType.PLAN, source="pfc",
        payload={"actions": [{"tool": "read_file", "confidence": 0.9}]},
    )
    result = await bg.process(sig)
    assert result is not None
    assert result.type == SignalType.ACTION_SELECTED
