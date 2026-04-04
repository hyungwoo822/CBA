import pytest
from brain_agent.regions.acc import AnteriorCingulateCortex
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def acc(): return AnteriorCingulateCortex()

async def test_no_conflict_passes_through(acc):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"actions": [{"tool": "read_file"}]})
    result = await acc.process(sig)
    assert result is None or result.type != SignalType.CONFLICT_DETECTED

async def test_error_accumulation(acc):
    for _ in range(5):
        sig = Signal(type=SignalType.ACTION_RESULT, source="cerebellum", payload={"expected": 1.0, "actual": 0.2})
        await acc.process(sig)
    assert acc.error_accumulator > 0

async def test_strategy_switch_on_threshold(acc):
    acc.strategy_switch_threshold = 0.5
    result = None
    for _ in range(10):
        sig = Signal(type=SignalType.ACTION_RESULT, source="cerebellum", payload={"expected": 1.0, "actual": 0.0})
        result = await acc.process(sig)
        if result and result.type == SignalType.STRATEGY_SWITCH: break
    assert result is not None
    assert result.type == SignalType.STRATEGY_SWITCH


async def test_patience_raises_conflict_threshold(acc):
    """Higher patience should raise the effective conflict threshold so the
    same conflict score no longer triggers CONFLICT_DETECTED (Doya 2002)."""
    actions = [{"tool": "read_file", "confidence": 0.05, "risk": 0.8}]

    # Low patience (0.1) => effective threshold = 0.6 * 0.6 = 0.36
    sig_low = Signal(
        type=SignalType.PLAN, source="pfc",
        payload={"actions": actions},
        metadata={"neuromodulators": {"patience": 0.1}},
    )
    result_low = acc._evaluate_plan(sig_low)

    # High patience (1.0) => effective threshold = 0.6 * 1.5 = 0.90
    sig_high = Signal(
        type=SignalType.PLAN, source="pfc",
        payload={"actions": actions},
        metadata={"neuromodulators": {"patience": 1.0}},
    )
    result_high = acc._evaluate_plan(sig_high)

    # Same conflict_score, but high patience should tolerate it
    # Low patience should detect conflict
    assert result_low is not None
    assert result_low.type == SignalType.CONFLICT_DETECTED
    # High patience may or may not detect — but if it does, it should be
    # at a higher threshold. We check the effective threshold logic:
    # With these specific values the conflict_score is:
    #   low_conf: 0.3-0.05 = 0.25  +  high_risk: 0.8*0.5 = 0.4  = 0.65
    # effective_threshold(patience=0.1) = 0.6*0.6 = 0.36  => 0.65 >= 0.36 => conflict
    # effective_threshold(patience=1.0) = 0.6*1.5 = 0.90  => 0.65 < 0.90  => no conflict
    assert result_high is None


async def test_patience_raises_strategy_switch_threshold():
    """Higher patience raises effective strategy switch threshold,
    requiring more accumulated error before switching (Doya 2002)."""
    # With low patience, strategy switch should happen sooner
    acc_low = AnteriorCingulateCortex()
    acc_low.strategy_switch_threshold = 1.0
    switches_low = 0
    for _ in range(5):
        sig = Signal(
            type=SignalType.ACTION_RESULT, source="cerebellum",
            payload={"expected": 1.0, "actual": 0.0},
            metadata={"neuromodulators": {"patience": 0.0}},
        )
        result = await acc_low.process(sig)
        if result and result.type == SignalType.STRATEGY_SWITCH:
            switches_low += 1

    # With high patience, strategy switch should take longer
    acc_high = AnteriorCingulateCortex()
    acc_high.strategy_switch_threshold = 1.0
    switches_high = 0
    for _ in range(5):
        sig = Signal(
            type=SignalType.ACTION_RESULT, source="cerebellum",
            payload={"expected": 1.0, "actual": 0.0},
            metadata={"neuromodulators": {"patience": 1.0}},
        )
        result = await acc_high.process(sig)
        if result and result.type == SignalType.STRATEGY_SWITCH:
            switches_high += 1

    # Low patience should cause more strategy switches than high patience
    assert switches_low >= switches_high
