import pytest
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.regions.wernicke import WernickeArea
from brain_agent.regions.amygdala import Amygdala
from brain_agent.regions.prefrontal import PrefrontalCortex
from brain_agent.regions.broca import BrocaArea


class TestWernickeInject:
    def test_inject_sets_comprehension_and_activation(self):
        w = WernickeArea(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "hello"})
        comprehension = {
            "intent": "greeting", "complexity": "simple",
            "keywords": ["hello"], "language": "en",
        }
        result = w.inject(signal, comprehension)
        assert result.payload["comprehension"] == comprehension
        assert w.activation_level > 0

    def test_inject_complex_raises_activation(self):
        w = WernickeArea(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "x"})
        w.inject(signal, {"intent": "greeting", "complexity": "simple", "keywords": []})
        act_simple = w.activation_level
        w.inject(signal, {"intent": "request", "complexity": "complex", "keywords": ["a", "b"]})
        act_complex = w.activation_level
        assert act_complex > act_simple


class TestAmygdalaInject:
    def test_inject_sets_emotional_tag(self):
        a = Amygdala(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "sad"})
        appraisal = {"valence": -0.5, "arousal": 0.6, "threat_detected": False, "primary_emotion": "sadness"}
        result = a.inject(signal, appraisal)
        assert result.emotional_tag is not None
        assert result.emotional_tag.valence == -0.5
        assert result.emotional_tag.arousal == 0.6
        assert result.metadata["amygdala_right"]["valence"] == -0.5
        assert result.metadata["amygdala_left"]["primary_emotion"] == "sadness"
        assert a.activation_level > 0

    def test_inject_threat_raises_activation(self):
        a = Amygdala(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "x"})
        a.inject(signal, {"valence": 0.0, "arousal": 0.1, "threat_detected": False, "primary_emotion": "neutral"})
        act_calm = a.activation_level
        a.inject(signal, {"valence": -0.8, "arousal": 0.9, "threat_detected": True, "primary_emotion": "fear"})
        act_threat = a.activation_level
        assert act_threat > act_calm


class TestPFCCorticalIntegration:
    def test_build_cortical_system_prompt_includes_instructions(self):
        pfc = PrefrontalCortex(llm_provider=None)
        prompt = pfc.build_cortical_system_prompt(
            upstream_context={}, memory_context=[], network_mode="executive_control",
        )
        assert '"comprehension"' in prompt
        assert '"appraisal"' in prompt
        assert '"intent"' in prompt
        assert '"valence"' in prompt

    def test_parse_cortical_response_extracts_all_fields(self):
        pfc = PrefrontalCortex(llm_provider=None)
        raw = '''<cortical>
{"comprehension": {"intent": "question", "complexity": "simple", "keywords": ["weather"], "language": "ko"},
 "appraisal": {"valence": 0.1, "arousal": 0.2, "threat_detected": false, "primary_emotion": "neutral"}}
</cortical>

Here is my response to you.

<meta>{"confidence": 0.85}</meta>

<entities>
{"entities": ["weather"], "about_user": [], "knowledge": []}
</entities>'''

        result = pfc.parse_cortical_response(raw)
        assert result["response"] == "Here is my response to you."
        assert result["comprehension"]["intent"] == "question"
        assert result["appraisal"]["valence"] == 0.1
        assert result["entities"]["entities"] == ["weather"]
        assert result["confidence"] == 0.85

    def test_parse_cortical_response_handles_missing_tags(self):
        pfc = PrefrontalCortex(llm_provider=None)
        raw = "Just a plain response with no tags."
        result = pfc.parse_cortical_response(raw)
        assert result["response"] == "Just a plain response with no tags."
        assert result["comprehension"]["intent"] == "statement"
        assert result["appraisal"]["valence"] == 0.0
        assert result["confidence"] == 0.7


class TestBrocaInject:
    def test_inject_refined_updates_signal(self):
        b = BrocaArea(llm_provider=None)
        signal = Signal(
            type=SignalType.PLAN, source="pfc",
            payload={"actions": [{"tool": "respond", "args": {"text": "original"}}]},
        )
        b.inject_refined(signal, "polished response")
        assert signal.payload["actions"][0]["args"]["text"] == "polished response"
        assert b.activation_level > 0

    def test_inject_refined_none_keeps_original(self):
        b = BrocaArea(llm_provider=None)
        signal = Signal(
            type=SignalType.PLAN, source="pfc",
            payload={"actions": [{"tool": "respond", "args": {"text": "original"}}]},
        )
        b.inject_refined(signal, None)
        assert signal.payload["actions"][0]["args"]["text"] == "original"
