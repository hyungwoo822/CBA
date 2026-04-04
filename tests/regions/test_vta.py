import pytest
from brain_agent.regions.vta import VentralTegmentalArea
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def vta():
    return VentralTegmentalArea()


def test_anatomy(vta):
    assert vta.lobe == Lobe.MIDBRAIN
    assert vta.hemisphere == Hemisphere.BILATERAL
    assert vta.position.x == 0
    assert vta.position.y == -20
    assert vta.position.z == -20


async def test_prediction_error_high_activation(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={"error": 0.9},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.9)
    assert result.metadata["vta_activation"] == pytest.approx(0.9)


async def test_prediction_error_small(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={"error": 0.1},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.1)
    assert result.metadata["vta_activation"] == pytest.approx(0.1)


async def test_prediction_error_clamped(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={"error": 2.5},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(1.0)


async def test_prediction_error_negative(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={"error": -0.7},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.7)


async def test_action_result_low_activation(vta):
    sig = Signal(
        type=SignalType.ACTION_RESULT,
        source="executor",
        payload={"result": "success"},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.2)
    assert result.metadata["vta_activation"] == pytest.approx(0.2)


async def test_other_signal_passthrough(vta):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    result = await vta.process(sig)
    assert result is sig
    assert vta.activation_level == pytest.approx(0.0)


async def test_zero_error(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={"error": 0.0},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.0)
    assert result.metadata["vta_activation"] == pytest.approx(0.0)


async def test_missing_error_field(vta):
    sig = Signal(
        type=SignalType.PREDICTION_ERROR,
        source="cerebellum",
        payload={},
    )
    result = await vta.process(sig)
    assert vta.activation_level == pytest.approx(0.0)
