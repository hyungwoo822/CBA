import pytest
from brain_agent.regions.cerebellum import Cerebellum
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def cere(): return Cerebellum()

async def test_predict_outcome(cere):
    sig = Signal(type=SignalType.ACTION_SELECTED, source="basal_ganglia", payload={"tool": "read_file"})
    result = await cere.process(sig)
    assert result is not None
    assert "predicted_outcome" in result.payload

async def test_small_error_no_escalation(cere):
    sig = Signal(type=SignalType.ACTION_RESULT, source="executor", payload={"error": 0.02})
    result = await cere.process(sig)
    assert result is None

async def test_large_error_escalates(cere):
    sig = Signal(type=SignalType.ACTION_RESULT, source="executor", payload={"error": 0.8, "predicted": "success", "actual": "failure"})
    result = await cere.process(sig)
    assert result is not None
    assert result.type == SignalType.PREDICTION_ERROR
