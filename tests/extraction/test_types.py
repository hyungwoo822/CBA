"""Tests for extraction dataclass contracts."""
from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.types import (
    ExtractionResult,
    SeverityDecision,
    TemporalResolveResult,
    TriageResult,
    ValidationResult,
)


def test_extraction_result_defaults():
    result = ExtractionResult(
        workspace_id="personal",
        source_id="src-1",
        narrative_chunk="hi",
    )
    assert result.nodes == []
    assert result.edges == []
    assert result.contradictions == []
    assert result.open_questions == []
    assert result.new_type_proposals == []
    assert result.response_text == ""
    assert result.response_mode == "normal"
    assert result.clarification_questions == []


def test_triage_result_defaults():
    result = TriageResult(target_workspace_id="personal", input_kinds=["spec_drop"])
    assert result.severity_hint == "none"
    assert result.skip_stages == []
    assert result.workspace_ask is None


def test_temporal_resolve_result_defaults():
    result = TemporalResolveResult()
    assert result.update_ops == []
    assert result.new_edges == []
    assert result.reinforced_edges == []


def test_validation_result_defaults():
    result = ValidationResult()
    assert result.contradictions == []
    assert result.open_questions == []


def test_severity_decision_defaults():
    result = SeverityDecision(response_mode="normal")
    assert result.clarification_questions == []


def test_extraction_result_carries_lists():
    result = ExtractionResult(
        workspace_id="w",
        source_id="s",
        narrative_chunk="x",
        nodes=[{"type": "Person", "label": "hyungpu"}],
        edges=[{"source": "hyungpu", "relation": "prefer", "target": "jjambbong"}],
        response_mode="append",
    )
    assert result.nodes[0]["label"] == "hyungpu"
    assert result.edges[0]["relation"] == "prefer"
    assert result.response_mode == "append"


def test_extraction_config_defaults():
    config = ExtractionConfig()
    assert config.triage_model == "auto"
    assert config.extract_model == "auto"
    assert config.temporal_classify_model == "auto"
    assert config.refine_model == "auto"
    assert config.max_retry == 1
    assert config.enable_severity_block is True
    assert config.promotion_threshold_n == 3
    assert config.max_open_questions_per_extraction == 3
    assert config.fok_similarity_threshold == 0.3
    assert config.pattern_separation_label_similarity_threshold == 0.75
    assert config.pattern_separation_window_hours == 24
