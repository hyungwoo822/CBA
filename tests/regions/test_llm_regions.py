"""Tests for LLM-enabled brain regions (Wernicke, Amygdala Left, Broca).

Uses a mock LLM provider to verify the LLM processing paths work correctly
without requiring an actual API connection.
"""
import pytest
import json
from dataclasses import dataclass, field

from brain_agent.regions.wernicke import WernickeArea
from brain_agent.regions.amygdala import Amygdala, AmygdalaLeft
from brain_agent.regions.broca import BrocaArea
from brain_agent.core.signals import Signal, SignalType, EmotionalTag


# ── Mock LLM Provider ─────────────────────────────────────────────────

@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class MockLLMProvider:
    """Mock LLM provider that returns predetermined responses."""

    def __init__(self, response_content: str):
        self._response = response_content
        self.call_count = 0
        self.last_messages = None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.call_count += 1
        self.last_messages = messages
        return MockLLMResponse(content=self._response)

    def get_default_model(self):
        return "mock-model"


class FailingLLMProvider:
    """Mock LLM provider that always raises an error."""

    async def chat(self, messages, **kwargs):
        raise ConnectionError("LLM API unreachable")

    def get_default_model(self):
        return "mock-model"


# ══════════════════════════════════════════════════════════════════════
# Wernicke LLM Tests
# ══════════════════════════════════════════════════════════════════════

class TestWernickeLLM:
    """Test Wernicke's area with LLM comprehension path."""

    @pytest.fixture
    def wernicke_llm(self):
        response = json.dumps({
            "intent": "question",
            "complexity": "moderate",
            "keywords": ["deploy", "production", "server"],
            "semantic_roles": {"agent": "user", "action": "deploy", "patient": "server"},
            "discourse_type": "technical",
            "language": "en",
        })
        provider = MockLLMProvider(response)
        return WernickeArea(llm_provider=provider), provider

    async def test_llm_comprehension(self, wernicke_llm):
        """Wernicke should use LLM for deep semantic analysis."""
        wernicke, provider = wernicke_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "How do I deploy to the production server?"},
        )
        result = await wernicke.process(sig)
        comp = result.payload["comprehension"]

        assert provider.call_count == 1
        assert comp["intent"] == "question"
        assert comp["complexity"] == "moderate"
        assert "deploy" in comp["keywords"]
        assert comp.get("semantic_roles") is not None
        assert comp.get("discourse_type") == "technical"

    async def test_llm_preserves_word_count(self, wernicke_llm):
        """LLM response should include backward-compatible word_count."""
        wernicke, _ = wernicke_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "How do I deploy?"},
        )
        result = await wernicke.process(sig)
        comp = result.payload["comprehension"]
        assert "word_count" in comp
        assert comp["word_count"] == 4

    async def test_llm_failure_falls_back_to_structural(self):
        """On LLM failure, Wernicke should fall back to structural parse."""
        wernicke = WernickeArea(llm_provider=FailingLLMProvider())
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "What is the meaning of life?"},
        )
        result = await wernicke.process(sig)
        comp = result.payload["comprehension"]
        # Structural parse: intent is always "statement" (no keyword faking)
        assert comp["intent"] == "statement"
        assert comp["word_count"] == 6

    async def test_llm_malformed_response_falls_back(self):
        """Malformed LLM response should trigger structural fallback."""
        wernicke = WernickeArea(llm_provider=MockLLMProvider("not json at all"))
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "run the tests"},
        )
        result = await wernicke.process(sig)
        comp = result.payload["comprehension"]
        assert comp["intent"] == "statement"  # structural fallback


# ══════════════════════════════════════════════════════════════════════
# Amygdala LLM Tests
# ══════════════════════════════════════════════════════════════════════

class TestAmygdalaLLM:
    """Test Amygdala with LLM-enabled left hemisphere."""

    @pytest.fixture
    def amygdala_llm(self):
        response = json.dumps({
            "valence": -0.3,
            "arousal": 0.6,
            "threat_level": "moderate",
            "primary_emotion": "fear",
            "contextual_factors": {
                "is_hypothetical": False,
                "is_sarcastic": False,
                "is_venting": False,
                "urgency": "high",
            },
        })
        provider = MockLLMProvider(response)
        return Amygdala(llm_provider=provider), provider

    async def test_right_uses_llm(self, amygdala_llm):
        """Right amygdala should use LLM for fast appraisal."""
        amygdala, provider = amygdala_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "critical error occurred"},
        )
        result = await amygdala.process(sig)
        r_data = result.metadata["amygdala_right"]
        assert "threat_detected" in r_data

    async def test_left_uses_llm(self, amygdala_llm):
        """Left amygdala should use LLM for contextual evaluation."""
        amygdala, provider = amygdala_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "The error handling code looks great"},
        )
        result = await amygdala.process(sig)
        # R + L both call LLM = 2 calls
        assert provider.call_count == 2
        l_data = result.metadata["amygdala_left"]
        assert "valence" in l_data
        assert l_data.get("primary_emotion") == "fear"

    async def test_bilateral_blend_with_llm(self, amygdala_llm):
        """Bilateral blend should work with LLM left result."""
        amygdala, _ = amygdala_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "something worrying happened"},
        )
        result = await amygdala.process(sig)
        assert result.emotional_tag is not None
        blend = result.metadata["amygdala_blend"]
        assert "right_weight" in blend
        assert "left_weight" in blend

    async def test_llm_failure_falls_back(self):
        """On LLM failure, left amygdala should use heuristic."""
        amygdala = Amygdala(llm_provider=FailingLLMProvider())
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"text": "please read file.py"},
        )
        result = await amygdala.process(sig)
        assert result.emotional_tag is not None
        assert result.emotional_tag.arousal < 0.5


# ══════════════════════════════════════════════════════════════════════
# Broca LLM Tests
# ══════════════════════════════════════════════════════════════════════

class TestBrocaLLM:
    """Test Broca's area with LLM language production path."""

    @pytest.fixture
    def broca_llm(self):
        provider = MockLLMProvider("Here is a well-formatted response about deployment.")
        return BrocaArea(llm_provider=provider), provider

    async def test_llm_production(self, broca_llm):
        """Broca should use LLM for language formulation."""
        broca, provider = broca_llm
        sig = Signal(
            type=SignalType.PLAN,
            source="prefrontal_cortex",
            payload={
                "actions": [{"tool": "respond", "args": {"text": "raw PFC output about deployment"}}],
            },
        )
        result = await broca.process(sig)
        assert provider.call_count == 1
        # LLM output should be applied
        assert result.payload["actions"][0]["args"]["text"] == "Here is a well-formatted response about deployment."

    async def test_llm_receives_context(self, broca_llm):
        """Broca LLM should receive comprehension and emotional context."""
        broca, provider = broca_llm
        sig = Signal(
            type=SignalType.PLAN,
            source="prefrontal_cortex",
            payload={
                "actions": [{"tool": "respond", "args": {"text": "some response"}}],
            },
        )
        sig.emotional_tag = EmotionalTag(valence=0.5, arousal=0.3)
        sig.metadata["comprehension"] = {"intent": "question", "language": "en"}
        sig.metadata["production_plan"] = {"register": "formal", "emphasis": ["deploy"]}

        await broca.process(sig)
        # Check that the LLM received context
        user_msg = provider.last_messages[-1]["content"]
        assert "valence" in user_msg or "Emotional" in user_msg

    async def test_llm_failure_falls_back(self):
        """On LLM failure, Broca should fall back to basic formatting."""
        broca = BrocaArea(llm_provider=FailingLLMProvider())
        sig = Signal(
            type=SignalType.PLAN,
            source="prefrontal_cortex",
            payload={
                "actions": [{"tool": "respond", "args": {"text": "  some   text  "}}],
            },
        )
        result = await broca.process(sig)
        assert result.payload["actions"][0]["args"]["text"] == "some text"

    async def test_ignores_non_plan(self, broca_llm):
        """Broca should pass through non-PLAN signals unchanged."""
        broca, provider = broca_llm
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="test",
            payload={"text": "hello"},
        )
        await broca.process(sig)
        assert provider.call_count == 0
