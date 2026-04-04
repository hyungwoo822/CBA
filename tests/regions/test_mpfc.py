"""Tests for MedialPrefrontalCortex — self-referential processing region.

References:
  - Northoff et al. (2006): Self-referential processing in mPFC
  - Ghosh & Gilboa (2014): Schema theory
  - D'Argembeau et al. (2005): mPFC and self-relevant thinking
"""

import pytest

from brain_agent.regions.mpfc import MedialPrefrontalCortex
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def mpfc():
    return MedialPrefrontalCortex()


def test_mpfc_basic_properties(mpfc):
    """Region has correct anatomical properties (name, lobe, hemisphere, position)."""
    assert mpfc.name == "medial_pfc"
    assert mpfc.lobe == Lobe.FRONTAL
    assert mpfc.hemisphere == Hemisphere.BILATERAL
    assert mpfc.position.x == 0
    assert mpfc.position.y == 45
    assert mpfc.position.z == 30


def test_mpfc_has_self_model(mpfc):
    """get_self_model() returns a dict with 'schema' and 'facts' keys."""
    model = mpfc.get_self_model()
    assert isinstance(model, dict)
    assert "schema" in model
    assert "facts" in model
    assert isinstance(model["schema"], str)
    assert isinstance(model["facts"], list)


def test_mpfc_loads_identity_from_file(mpfc):
    """get_self_context() returns non-empty string loaded from SOUL.md."""
    context = mpfc.get_self_context()
    assert isinstance(context, str)
    assert len(context) > 0
    # Should contain content from SOUL.md
    assert "Soul" in context or "identity" in context.lower() or "neural" in context.lower()


def test_mpfc_merges_schema_and_graph(mpfc):
    """Graph facts injected via update_from_graph_facts appear in get_self_context()."""
    facts = [
        {"subject": "agent", "predicate": "prefers", "object": "concise responses"},
        {"subject": "agent", "predicate": "values", "object": "honesty"},
    ]
    mpfc.update_from_graph_facts(facts)
    context = mpfc.get_self_context()
    assert "concise responses" in context
    assert "honesty" in context


async def test_mpfc_process_returns_signal(mpfc):
    """process() returns a signal with self_context in metadata."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Who are you?"},
    )
    result = await mpfc.process(sig)
    assert result is not None
    assert isinstance(result, Signal)
    assert "self_context" in result.metadata
    assert len(result.metadata["self_context"]) > 0


async def test_mpfc_process_emits_activation(mpfc):
    """process() sets activation level to 0.4."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Tell me about yourself"},
    )
    await mpfc.process(sig)
    assert mpfc.activation_level == pytest.approx(0.4)


def test_mpfc_update_from_graph_facts_stores_facts(mpfc):
    """update_from_graph_facts() stores the provided facts internally."""
    facts = [{"subject": "self", "predicate": "is", "object": "curious"}]
    mpfc.update_from_graph_facts(facts)
    model = mpfc.get_self_model()
    assert model["facts"] == facts


def test_mpfc_schema_text_loaded_on_init(mpfc):
    """Schema text is loaded from SOUL.md during initialization."""
    model = mpfc.get_self_model()
    assert len(model["schema"]) > 0
