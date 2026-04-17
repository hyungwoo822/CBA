import pytest
from unittest.mock import AsyncMock, MagicMock
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.regions.wernicke import WernickeArea
from brain_agent.regions.amygdala import Amygdala
from brain_agent.regions.prefrontal import PrefrontalCortex
from brain_agent.regions.broca import BrocaArea
from brain_agent.providers.base import LLMResponse


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


class TestExtractionOrchestratorParity:
    """Phase 5 orchestrator replaces the old PSC direct call."""

    @pytest.mark.asyncio
    async def test_orchestrator_returns_extraction_result(self):
        from brain_agent.config.schema import ExtractionConfig
        from brain_agent.extraction.orchestrator import ExtractionOrchestrator
        from brain_agent.memory.manager import MemoryManager

        mock_mem = MagicMock(spec=MemoryManager)
        mock_mem._interaction_counter = 0
        mock_mem.workspace = MagicMock()
        mock_mem.workspace.get_session_workspace = AsyncMock(return_value="personal")
        mock_mem.workspace.get_workspace = AsyncMock(return_value={"id": "personal", "name": "Personal Knowledge"})
        mock_mem.raw_vault = MagicMock()
        mock_mem.raw_vault.ingest = AsyncMock(return_value={"id": "src1"})
        mock_mem.ontology = MagicMock()
        mock_mem.ontology.get_node_types = AsyncMock(return_value=[{"name": "Concept"}])
        mock_mem.ontology.get_relation_types = AsyncMock(return_value=[{"name": "ask"}])
        mock_mem.ontology.get_node_schema = AsyncMock(return_value={"required": []})
        mock_mem.ontology.increment_occurrence = AsyncMock()
        mock_mem.ontology.register_node_type = AsyncMock()
        mock_mem.ontology.propose_node_type = AsyncMock()
        mock_mem.ontology.register_relation_type = AsyncMock()
        mock_mem.ontology.propose_relation_type = AsyncMock()
        mock_mem.semantic = MagicMock()
        mock_mem.semantic.get_relationships = AsyncMock(return_value=[])
        mock_mem.semantic.search = AsyncMock(return_value=[])
        mock_mem.semantic.find_events_near = AsyncMock(return_value=[])
        mock_mem.semantic.mark_superseded = AsyncMock()
        mock_mem.staging = MagicMock()
        mock_mem.staging.encode = AsyncMock(return_value="ep1")
        mock_mem.staging.encode_edge = AsyncMock()
        mock_mem.staging.reinforce = AsyncMock()
        mock_mem.contradictions = MagicMock()
        mock_mem.contradictions.detect = AsyncMock()
        mock_mem.open_questions = MagicMock()
        mock_mem.open_questions.add_question = AsyncMock()
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=LLMResponse(
            content='{"nodes":[{"type":"Concept","label":"weather","properties":{},"confidence":"EXTRACTED"}],'
                    '"edges":[{"source":"user","relation":"ask","target":"weather","confidence":"EXTRACTED",'
                    '"epistemic_source":"asserted","importance_score":0.8,"never_decay":0}],'
                    '"new_type_proposals":[],"narrative_chunk":"what is the weather"}',
        ))
        mock_provider.get_default_model.return_value = "mock-model"

        orchestrator = ExtractionOrchestrator(
            memory=mock_mem,
            llm_provider=mock_provider,
            config=ExtractionConfig(),
        )
        result = await orchestrator.extract(
            text="user ask weather important",
            session_id="s1",
            comprehension={"intent": "inform", "language": "en"},
        )

        assert result.workspace_id == "personal"
        assert result.response_mode in {"normal", "append", "block"}
        assert hasattr(result, "clarification_questions")
        assert result.edges[0]["target"] == "weather"
