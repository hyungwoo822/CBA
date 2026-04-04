"""Tests for Wernicke's Area — language comprehension.

Without LLM: returns structural parse only (word count, complexity).
Intent is always "statement" — no keyword-based fake classification.
With LLM (mock): returns full semantic analysis.
"""
import json
import pytest
from dataclasses import dataclass, field

from brain_agent.regions.wernicke import WernickeArea
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self):
        return len(self.tool_calls) > 0


class MockLLMProvider:
    def __init__(self, response_content: str):
        self._response = response_content

    async def chat(self, messages, **kwargs):
        return MockLLMResponse(content=self._response)

    def get_default_model(self):
        return "mock"


# ── Without LLM: structural parse only ────────────────────────────────

@pytest.fixture
def wernicke():
    return WernickeArea()


def test_anatomy(wernicke):
    assert wernicke.lobe == Lobe.TEMPORAL
    assert wernicke.hemisphere == Hemisphere.LEFT
    assert wernicke.position.x == -40


async def test_structural_parse_word_count(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke.process(sig)
    comp = result.payload["comprehension"]
    assert comp["word_count"] == 6


async def test_no_llm_intent_is_statement(wernicke):
    """Without LLM, intent is always 'statement' — no keyword faking."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke.process(sig)
    assert result.payload["comprehension"]["intent"] == "statement"


async def test_complexity_simple(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Hello world"})
    result = await wernicke.process(sig)
    assert result.payload["comprehension"]["complexity"] == "simple"


async def test_complexity_complex(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "If the server crashes and the database is corrupted because of a bug then we need to restart while monitoring logs although it might take time"})
    result = await wernicke.process(sig)
    assert result.payload["comprehension"]["complexity"] == "complex"


async def test_passthrough_no_text(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"image_data": b"data"})
    result = await wernicke.process(sig)
    assert "comprehension" not in result.payload


async def test_empty_text(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "   "})
    result = await wernicke.process(sig)
    assert "comprehension" not in result.payload


async def test_activation_simple(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "hello"})
    await wernicke.process(sig)
    assert wernicke.activation_level == pytest.approx(0.5)


async def test_activation_complex(wernicke):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "If the server crashes and the database fails because of bugs while we wait although it takes time and resources or perhaps not"})
    await wernicke.process(sig)
    assert wernicke.activation_level == pytest.approx(0.8)


async def test_keywords_are_empty_without_llm(wernicke):
    """Without LLM, keywords are empty (cannot extract without understanding)."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Find the authentication bug in the login module"})
    result = await wernicke.process(sig)
    assert result.payload["comprehension"]["keywords"] == []


# ── With LLM: full semantic analysis ──────────────────────────────────

@pytest.fixture
def wernicke_llm():
    response = json.dumps({
        "intent": "question",
        "complexity": "moderate",
        "keywords": ["meaning", "life"],
        "semantic_roles": {"agent": "user", "action": "ask", "topic": "meaning of life"},
        "discourse_type": "social",
        "language": "en",
    })
    return WernickeArea(llm_provider=MockLLMProvider(response))


async def test_llm_classifies_intent(wernicke_llm):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke_llm.process(sig)
    assert result.payload["comprehension"]["intent"] == "question"


async def test_llm_extracts_keywords(wernicke_llm):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke_llm.process(sig)
    assert "meaning" in result.payload["comprehension"]["keywords"]


async def test_llm_provides_semantic_roles(wernicke_llm):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke_llm.process(sig)
    assert result.payload["comprehension"]["semantic_roles"]["topic"] == "meaning of life"


async def test_llm_preserves_word_count(wernicke_llm):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "What is the meaning of life?"})
    result = await wernicke_llm.process(sig)
    assert result.payload["comprehension"]["word_count"] == 6
