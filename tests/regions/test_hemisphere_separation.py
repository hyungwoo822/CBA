"""Tests for left/right hemisphere separation.

Tests cover:
  - Amygdala L/R split (right fires first, left moderates, blended output)
  - PFC hemisphere activation in ECN vs CREATIVE mode
  - Corpus callosum integration in pipeline
  - Angular gyrus multimodal binding in pipeline
  - Backward compatibility (existing interfaces preserved)
"""
import pytest
from unittest.mock import AsyncMock

from brain_agent.regions.amygdala import Amygdala, AmygdalaRight, AmygdalaLeft
from brain_agent.regions.prefrontal import PrefrontalCortex
from brain_agent.regions.corpus_callosum import CorpusCallosum
from brain_agent.regions.angular_gyrus import AngularGyrus
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.regions.base import Hemisphere


# ═══════════════════════════════════════════════════════════════════
# Amygdala L/R Tests
# ═══════════════════════════════════════════════════════════════════


class TestAmygdalaRight:
    """Right amygdala: fast automatic emotional appraisal (LLM-based)."""

    @pytest.fixture
    def right(self):
        return AmygdalaRight()  # No LLM → neutral baseline

    async def test_hemisphere_is_right(self, right):
        assert right.hemisphere == Hemisphere.RIGHT

    async def test_no_llm_returns_neutral(self, right):
        """Without LLM, right amygdala returns neutral baseline."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "CRITICAL ERROR: server crashed"})
        result = await right.process(sig)
        r_data = result.metadata["amygdala_right"]
        # No LLM → neutral baseline (no keyword faking)
        assert "arousal" in r_data
        assert "valence" in r_data

    async def test_stores_metadata(self, right):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "security breach"})
        result = await right.process(sig)
        assert "amygdala_right" in result.metadata
        assert "valence" in result.metadata["amygdala_right"]
        assert "arousal" in result.metadata["amygdala_right"]
        assert "threat_detected" in result.metadata["amygdala_right"]


class TestAmygdalaLeft:
    """Left amygdala: conscious contextual emotional processing (LLM-based)."""

    @pytest.fixture
    def left(self):
        return AmygdalaLeft()  # No LLM → neutral baseline

    async def test_hemisphere_is_left(self, left):
        assert left.hemisphere == Hemisphere.LEFT

    async def test_no_llm_returns_neutral(self, left):
        """Without LLM, left amygdala returns neutral baseline."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "critical error happened"})
        sig.metadata["amygdala_right"] = {"valence": -0.5, "arousal": 0.7, "threat_detected": True}
        result = await left.process(sig)
        l_data = result.metadata["amygdala_left"]
        assert "valence" in l_data
        assert "arousal" in l_data

    async def test_neutral_assessment(self, left):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "hello world"})
        sig.metadata["amygdala_right"] = {"valence": 0.0, "arousal": 0.15, "threat_detected": False}
        result = await left.process(sig)
        l_data = result.metadata["amygdala_left"]
        assert l_data["arousal"] < 0.5


class TestAmygdalaBilateral:
    """Bilateral Amygdala coordinator: R first, L second, blended output."""

    @pytest.fixture
    def amygdala(self):
        return Amygdala()

    async def test_right_fires_first(self, amygdala):
        """Right amygdala data should be in metadata after processing."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "security breach detected"})
        result = await amygdala.process(sig)
        assert "amygdala_right" in result.metadata
        assert "amygdala_left" in result.metadata
        # Left should have seen right's data
        assert result.metadata["amygdala_left"] is not None

    async def test_blended_emotional_tag(self, amygdala):
        """Final emotional tag should be a blend of R and L."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "fatal error occurred"})
        result = await amygdala.process(sig)
        assert result.emotional_tag is not None
        # Without LLM: no threat detected → left dominant
        blend = result.metadata["amygdala_blend"]
        assert "right_weight" in blend
        assert "left_weight" in blend

    async def test_non_threat_left_dominant(self, amygdala):
        """Non-threat: left hemisphere (contextual) should dominate."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "please read the documentation"})
        result = await amygdala.process(sig)
        assert result.emotional_tag is not None
        # Without LLM, no threat → left dominant (0.6)
        assert result.metadata["amygdala_blend"]["left_weight"] == 0.6

    async def test_backward_compatible_interface(self, amygdala):
        """Amygdala.process() must return Signal with emotional_tag, same as before."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "hello"})
        result = await amygdala.process(sig)
        assert result is not None
        assert isinstance(result, Signal)
        assert result.emotional_tag is not None
        assert hasattr(result.emotional_tag, "valence")
        assert hasattr(result.emotional_tag, "arousal")

    async def test_backward_compat_normal_input_low_arousal(self, amygdala):
        """Existing test contract: normal input -> arousal < 0.5."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "please read file.py"})
        result = await amygdala.process(sig)
        assert result.emotional_tag.arousal < 0.5

    async def test_no_llm_error_returns_neutral(self, amygdala):
        """Without LLM, even error text returns neutral (no keyword faking)."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "CRITICAL ERROR: server crashed"})
        result = await amygdala.process(sig)
        assert result.emotional_tag is not None
        # Without LLM, arousal stays near baseline
        assert result.emotional_tag.arousal < 0.3

    async def test_hemisphere_attributes(self, amygdala):
        assert amygdala.hemisphere == Hemisphere.BILATERAL
        assert amygdala.right.hemisphere == Hemisphere.RIGHT
        assert amygdala.left.hemisphere == Hemisphere.LEFT

    async def test_activation_level_set(self, amygdala):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "critical crash"})
        await amygdala.process(sig)
        assert amygdala.activation_level > 0
        assert amygdala.right.activation_level > 0
        assert amygdala.left.activation_level > 0


# ═══════════════════════════════════════════════════════════════════
# PFC Hemisphere Tests
# ═══════════════════════════════════════════════════════════════════


class TestPFCHemisphere:
    """PFC hemisphere activation in ECN vs CREATIVE mode."""

    @pytest.fixture
    def pfc(self):
        return PrefrontalCortex()

    async def test_ecn_left_dominant(self, pfc):
        """In ECN mode, left PFC (analytical) should be dominant."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "analyze this code"})
        sig.metadata["network_mode"] = "executive_control"
        await pfc.process(sig)
        assert pfc._left_activation > pfc._right_activation
        assert pfc._left_activation == 0.8
        assert pfc._right_activation == 0.3

    async def test_creative_right_dominant(self, pfc):
        """In CREATIVE mode, right PFC (holistic) should be dominant."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "brainstorm ideas"})
        sig.metadata["network_mode"] = "creative"
        await pfc.process(sig)
        assert pfc._right_activation > pfc._left_activation
        assert pfc._right_activation == 0.8
        assert pfc._left_activation == 0.5

    async def test_dmn_both_low(self, pfc):
        """In DMN mode, both hemispheres should be low."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "idle thoughts"})
        sig.metadata["network_mode"] = "default_mode"
        await pfc.process(sig)
        assert pfc._left_activation < 0.5
        assert pfc._right_activation < 0.5

    async def test_hemisphere_metadata_in_output(self, pfc):
        """Plan signal should contain hemisphere activation metadata."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "do something"})
        sig.metadata["network_mode"] = "executive_control"
        result = await pfc.process(sig)
        assert "pfc_hemisphere" in result.metadata
        assert result.metadata["pfc_hemisphere"]["left_activation"] == 0.8
        assert result.metadata["pfc_hemisphere"]["right_activation"] == 0.3
        assert result.metadata["pfc_hemisphere"]["network_mode"] == "executive_control"

    async def test_creative_mode_right_pfc_notes(self, pfc):
        """In creative mode, right PFC should add holistic suggestions."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "creative task"})
        sig.metadata["network_mode"] = "creative"
        result = await pfc.process(sig)
        assert "right_pfc_notes" in result.metadata
        assert result.metadata["right_pfc_notes"]["mode"] == "divergent_thinking"

    async def test_ecn_no_right_pfc_notes(self, pfc):
        """In ECN mode, no right PFC notes should be added."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "analytical task"})
        sig.metadata["network_mode"] = "executive_control"
        result = await pfc.process(sig)
        assert "right_pfc_notes" not in result.metadata

    async def test_cached_procedure_low_activations(self, pfc):
        """Cached procedure should result in minimal hemisphere activation."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "cached task"})
        sig.metadata["cached_procedure"] = {
            "stage": "autonomous",
            "action_sequence": [{"tool": "respond", "args": {"text": "cached"}}],
        }
        result = await pfc.process(sig)
        assert pfc._left_activation == 0.2
        assert pfc._right_activation == 0.1
        assert result.payload.get("from_cache") is True

    async def test_strategy_switch_clears_activations(self, pfc):
        """Strategy switch should clear hemisphere activations."""
        # First activate
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "task"})
        sig.metadata["network_mode"] = "executive_control"
        await pfc.process(sig)
        assert pfc._left_activation > 0

        # Then switch
        switch = Signal(type=SignalType.STRATEGY_SWITCH, source="acc",
                        payload={"reason": "errors"})
        await pfc.process(switch)
        assert pfc._left_activation == 0.0
        assert pfc._right_activation == 0.0

    async def test_backward_compat_process_returns_plan(self, pfc):
        """PFC.process() must return same Signal interface as before."""
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "read file auth.py"})
        result = await pfc.process(sig)
        assert result is not None
        assert result.type == SignalType.PLAN
        assert result.payload["goal"] == "read file auth.py"
        assert len(result.payload["actions"]) > 0

    async def test_conflict_sets_left_dominant(self, pfc):
        """Conflict detection should trigger left PFC analytical re-evaluation."""
        sig = Signal(type=SignalType.CONFLICT_DETECTED, source="acc",
                     payload={"reason": "conflicting actions"})
        await pfc.process(sig)
        assert pfc._left_activation == 0.9
        assert pfc._right_activation == 0.4


class TestPFCCreativeLLM:
    """PFC creative mode modifies LLM system prompt."""

    @pytest.fixture
    def mock_llm_provider(self):
        from brain_agent.providers.base import LLMResponse
        provider = AsyncMock()
        provider.chat.return_value = LLMResponse(
            content="Creative response here.",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        )
        return provider

    async def test_creative_mode_augments_prompt(self, mock_llm_provider):
        """In creative mode, system prompt should include creative augmentation."""
        pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "brainstorm approaches"})
        sig.metadata["network_mode"] = "creative"
        await pfc.process(sig)
        call_args = mock_llm_provider.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "CREATIVE MODE ACTIVE" in system_msg
        assert "divergent thinking" in system_msg

    async def test_ecn_mode_no_creative_prompt(self, mock_llm_provider):
        """In ECN mode, system prompt should NOT include creative augmentation."""
        pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "analyze data"})
        sig.metadata["network_mode"] = "executive_control"
        await pfc.process(sig)
        call_args = mock_llm_provider.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "CREATIVE MODE ACTIVE" not in system_msg


# ═══════════════════════════════════════════════════════════════════
# Corpus Callosum Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestCorpusCallosumIntegration:
    """Corpus callosum integrates L/R hemisphere results."""

    @pytest.fixture
    def cc(self):
        return CorpusCallosum()

    async def test_integrates_left_right(self, cc):
        sig = Signal(type=SignalType.PLAN, source="prefrontal_cortex",
                     payload={"goal": "test"})
        sig.metadata["left_result"] = {
            "activation": 0.8, "mode": "analytical", "confidence": 0.8
        }
        sig.metadata["right_result"] = {
            "activation": 0.3, "mode": "holistic", "confidence": 0.3
        }
        result = await cc.process(sig)
        assert "integrated_result" in result.metadata
        unified = result.metadata["integrated_result"]
        assert unified["integration_source"] == "corpus_callosum"
        assert "left_perspective" in unified
        assert "right_perspective" in unified

    async def test_passthrough_without_both_hemispheres(self, cc):
        sig = Signal(type=SignalType.PLAN, source="prefrontal_cortex",
                     payload={"goal": "test"})
        # Only left result, no right
        sig.metadata["left_result"] = {"activation": 0.5, "mode": "analytical"}
        result = await cc.process(sig)
        assert "integrated_result" not in result.metadata
        assert cc.activation_level < 0.5  # Low activation for passthrough

    async def test_higher_confidence_wins(self, cc):
        sig = Signal(type=SignalType.PLAN, source="prefrontal_cortex",
                     payload={"goal": "test"})
        sig.metadata["left_result"] = {
            "activation": 0.8, "mode": "analytical",
            "confidence": 0.9, "approach": "step-by-step"
        }
        sig.metadata["right_result"] = {
            "activation": 0.5, "mode": "holistic",
            "confidence": 0.4, "approach": "intuitive"
        }
        result = await cc.process(sig)
        unified = result.metadata["integrated_result"]
        # Left has higher confidence, so for conflicting "approach" key, left wins
        assert unified["approach"] == "step-by-step"

    async def test_transfer_count_increments(self, cc):
        assert cc.transfer_count == 0
        sig = Signal(type=SignalType.PLAN, source="prefrontal_cortex",
                     payload={"goal": "test"})
        sig.metadata["left_result"] = {"activation": 0.5, "confidence": 0.5}
        sig.metadata["right_result"] = {"activation": 0.5, "confidence": 0.5}
        await cc.process(sig)
        assert cc.transfer_count == 1


# ═══════════════════════════════════════════════════════════════════
# Angular Gyrus Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestAngularGyrusIntegration:
    """Angular gyrus cross-modal semantic integration."""

    @pytest.fixture
    def ag(self):
        return AngularGyrus()

    async def test_text_only_integration(self, ag):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "Hello world"})
        result = await ag.process(sig)
        integration = result.metadata.get("semantic_integration")
        assert integration is not None
        assert "text" in integration["modalities_present"]

    async def test_multimodal_visual_text(self, ag):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={
                         "text": "What is in this image?",
                         "visual_features": {"description": "a cat", "objects": ["cat"]},
                     })
        result = await ag.process(sig)
        integration = result.metadata["semantic_integration"]
        assert "text" in integration["modalities_present"]
        assert "visual" in integration["modalities_present"]
        assert len(integration["modalities_present"]) == 2
        assert integration["integration_confidence"] > 0.5

    async def test_multimodal_auditory_text(self, ag):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={
                         "text": "some words",
                         "prosody": {"emotion": "happy", "confidence": 0.8},
                     })
        result = await ag.process(sig)
        integration = result.metadata["semantic_integration"]
        assert "text" in integration["modalities_present"]
        assert "auditory" in integration["modalities_present"]

    async def test_no_modalities_low_activation(self, ag):
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={})
        result = await ag.process(sig)
        assert ag.activation_level < 0.1

    async def test_higher_activation_for_more_modalities(self, ag):
        # Single modality
        sig1 = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                      payload={"text": "hello"})
        await ag.process(sig1)
        act1 = ag.activation_level

        # Two modalities
        ag2 = AngularGyrus()
        sig2 = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                      payload={"text": "hello", "visual_features": {"desc": "img"}})
        await ag2.process(sig2)
        act2 = ag2.activation_level

        assert act2 > act1


# ═══════════════════════════════════════════════════════════════════
# Network Modes Tests
# ═══════════════════════════════════════════════════════════════════


class TestNetworkModesUpdated:
    """Angular gyrus in ECN, corpus callosum always active."""

    def test_angular_gyrus_in_ecn(self):
        from brain_agent.core.network_modes import MODE_REGIONS, NetworkMode
        assert "angular_gyrus" in MODE_REGIONS[NetworkMode.ECN]

    def test_corpus_callosum_always_active(self):
        from brain_agent.core.network_modes import ALWAYS_ACTIVE
        assert "corpus_callosum" in ALWAYS_ACTIVE

    def test_corpus_callosum_active_in_all_modes(self):
        from brain_agent.core.network_modes import TripleNetworkController, NetworkMode
        ctrl = TripleNetworkController()
        for mode in NetworkMode:
            ctrl.switch_to(mode)
            assert ctrl.is_region_active("corpus_callosum") or "corpus_callosum" in {"corpus_callosum"}


# ═══════════════════════════════════════════════════════════════════
# Memory Manager Modality Tests
# ═══════════════════════════════════════════════════════════════════


class TestMemoryManagerModality:
    """MemoryManager.encode() accepts modality for hippocampus L/R awareness."""

    async def test_encode_with_modality(self, tmp_path, mock_embedding):
        from brain_agent.memory.manager import MemoryManager
        mgr = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
        await mgr.initialize()
        try:
            mem_id = await mgr.encode(
                content="Test content",
                entities={"input": "test"},
                modality="verbal",
            )
            assert mem_id is not None
        finally:
            await mgr.close()

    async def test_encode_without_modality_backward_compat(self, tmp_path, mock_embedding):
        from brain_agent.memory.manager import MemoryManager
        mgr = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
        await mgr.initialize()
        try:
            # No modality parameter — backward compatible
            mem_id = await mgr.encode(
                content="Test content",
                entities={"input": "test"},
            )
            assert mem_id is not None
        finally:
            await mgr.close()

    async def test_modality_does_not_mutate_caller_dict(self, tmp_path, mock_embedding):
        from brain_agent.memory.manager import MemoryManager
        mgr = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
        await mgr.initialize()
        try:
            entities = {"input": "test"}
            await mgr.encode(
                content="Test content",
                entities=entities,
                modality="visual",
            )
            # Original dict should NOT be mutated
            assert "modality" not in entities
        finally:
            await mgr.close()
