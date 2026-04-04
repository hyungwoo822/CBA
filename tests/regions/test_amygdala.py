"""Tests for Amygdala — bilateral emotional processing.

Without LLM, amygdala returns neutral baselines (no fake keyword emotion).
With LLM (mock), it performs genuine emotional evaluation.
"""
import json
import pytest
from dataclasses import dataclass, field

from brain_agent.regions.amygdala import Amygdala
from brain_agent.core.signals import Signal, SignalType


@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ThreatDetectingMockLLM:
    """Mock LLM that returns high arousal for threat-like inputs."""

    async def chat(self, messages, **kwargs):
        user_msg = messages[-1]["content"].lower()
        # Right amygdala prompt → quick assessment
        if "fast" in messages[0].get("content", "").lower() or "quick" in messages[0].get("content", "").lower():
            if any(w in user_msg for w in ["error", "crash", "critical", "breach", "security"]):
                return MockLLMResponse(content=json.dumps({
                    "valence": -0.7, "arousal": 0.85, "threat_detected": True,
                }))
            return MockLLMResponse(content=json.dumps({
                "valence": 0.0, "arousal": 0.15, "threat_detected": False,
            }))
        # Left amygdala prompt → contextual
        if any(w in user_msg for w in ["error", "crash", "critical", "breach", "security"]):
            return MockLLMResponse(content=json.dumps({
                "valence": -0.6, "arousal": 0.8, "threat_level": "high",
                "primary_emotion": "fear",
                "contextual_factors": {"is_hypothetical": False, "urgency": "high"},
            }))
        return MockLLMResponse(content=json.dumps({
            "valence": 0.0, "arousal": 0.1, "threat_level": "none",
            "primary_emotion": "neutral",
            "contextual_factors": {"urgency": "none"},
        }))

    def get_default_model(self):
        return "mock"


# ── Without LLM: neutral baselines ────────────────────────────────────

@pytest.fixture
def amygdala_no_llm():
    return Amygdala()


async def test_no_llm_returns_neutral(amygdala_no_llm):
    """Without LLM, amygdala returns neutral baselines (no keyword faking)."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "CRITICAL ERROR: server crashed"})
    result = await amygdala_no_llm.process(sig)
    assert result.emotional_tag is not None
    # Without LLM, should be near-neutral (no keyword detection)
    assert result.emotional_tag.arousal < 0.3


async def test_no_llm_always_has_tag(amygdala_no_llm):
    """Without LLM, still produces an emotional tag (just neutral)."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "please read file.py"})
    result = await amygdala_no_llm.process(sig)
    assert result.emotional_tag is not None


# ── With LLM: genuine emotional evaluation ────────────────────────────

@pytest.fixture
def amygdala_llm():
    return Amygdala(llm_provider=ThreatDetectingMockLLM())


async def test_llm_detects_threat(amygdala_llm):
    """With LLM, amygdala should detect threat in error messages."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "CRITICAL ERROR: server crashed"})
    result = await amygdala_llm.process(sig)
    assert result.emotional_tag.arousal > 0.5
    assert result.emotional_tag.valence < 0


async def test_llm_security_breach(amygdala_llm):
    """With LLM, amygdala should detect high threat for security breach."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "security breach detected"})
    result = await amygdala_llm.process(sig)
    assert result.emotional_tag.arousal > 0.5


async def test_llm_neutral_input(amygdala_llm):
    """With LLM, neutral input should get low arousal."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "please read file.py"})
    result = await amygdala_llm.process(sig)
    assert result.emotional_tag.arousal < 0.5


async def test_bilateral_blend_metadata(amygdala_llm):
    """Should produce blend metadata from both hemispheres."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "something happened"})
    result = await amygdala_llm.process(sig)
    assert "amygdala_right" in result.metadata
    assert "amygdala_left" in result.metadata
    assert "amygdala_blend" in result.metadata
