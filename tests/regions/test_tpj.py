"""Tests for TemporoparietalJunction (Theory of Mind) region."""
import pytest
from brain_agent.regions.tpj import TemporoparietalJunction
from brain_agent.regions.base import Hemisphere, Lobe
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def tpj():
    return TemporoparietalJunction()


# ── Basic properties ──────────────────────────────────────────────────

def test_tpj_basic_properties(tpj):
    assert tpj.name == "tpj"
    assert tpj.hemisphere == Hemisphere.RIGHT
    assert tpj.lobe == Lobe.PARIETAL
    assert tpj.position.x == 50
    assert tpj.position.y == -35
    assert tpj.position.z == 30


# ── User model ────────────────────────────────────────────────────────

def test_tpj_has_user_model(tpj):
    model = tpj.get_user_model()
    assert isinstance(model, dict)
    assert "schema" in model
    assert "facts" in model
    assert isinstance(model["schema"], str)
    assert isinstance(model["facts"], list)


# ── Profile loading ───────────────────────────────────────────────────

def test_tpj_loads_profile_from_file(tpj):
    context = tpj.get_user_context()
    assert isinstance(context, str)
    assert len(context) > 0
    # The USER.md template contains "User Profile"
    assert "User Profile" in context


# ── Schema + graph merge ──────────────────────────────────────────────

def test_tpj_merges_schema_and_graph(tpj):
    facts = [
        {"subject": "user", "predicate": "name", "object": "Alice"},
        {"subject": "user", "predicate": "occupation", "object": "engineer"},
    ]
    tpj.update_from_graph_facts(facts)
    context = tpj.get_user_context()
    assert "Alice" in context
    assert "engineer" in context


# ── Process returns signal with user_context ──────────────────────────

async def test_tpj_process_returns_signal(tpj):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Hello there"},
    )
    result = await tpj.process(sig)
    assert result is not None
    assert "user_context" in result.metadata
    assert isinstance(result.metadata["user_context"], str)
    assert tpj.activation_level == pytest.approx(0.4)
