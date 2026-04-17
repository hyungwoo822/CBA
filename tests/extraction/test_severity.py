"""Tests for Stage 4 Severity Branch."""
from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.severity import compute_response_mode


def test_no_issues_returns_normal():
    decision = compute_response_mode([], [], ExtractionConfig())
    assert decision.response_mode == "normal"
    assert decision.clarification_questions == []


def test_severe_contradiction_blocks():
    decision = compute_response_mode(
        [{"subject": "a", "key": "r", "value_a": "b", "value_b": "c", "severity": "severe"}],
        [],
        ExtractionConfig(),
    )
    assert decision.response_mode == "block"


def test_severe_question_blocks():
    decision = compute_response_mode(
        [],
        [{"question": "which?", "severity": "severe", "raised_by": "x"}],
        ExtractionConfig(),
    )
    assert decision.response_mode == "block"


def test_moderate_contradiction_appends():
    decision = compute_response_mode(
        [{"subject": "a", "key": "r", "value_a": "b", "value_b": "c", "severity": "moderate"}],
        [],
        ExtractionConfig(),
    )
    assert decision.response_mode == "append"


def test_moderate_question_appends():
    decision = compute_response_mode(
        [],
        [{"question": "q", "severity": "moderate", "raised_by": "x"}],
        ExtractionConfig(),
    )
    assert decision.response_mode == "append"


def test_minor_only_returns_append_for_contradiction():
    decision = compute_response_mode(
        [{"subject": "a", "key": "r", "value_a": "b", "value_b": "c", "severity": "minor"}],
        [],
        ExtractionConfig(),
    )
    assert decision.response_mode == "append"


def test_clarification_questions_populated():
    decision = compute_response_mode(
        [],
        [
            {"question": "Q1", "severity": "moderate", "raised_by": "x"},
            {"question": "Q2", "severity": "moderate", "raised_by": "y"},
        ],
        ExtractionConfig(),
    )
    assert decision.clarification_questions == ["Q1", "Q2"]


def test_block_can_be_disabled_via_config():
    config = ExtractionConfig(enable_severity_block=False)
    decision = compute_response_mode(
        [{"subject": "a", "key": "r", "value_a": "b", "value_b": "c", "severity": "severe"}],
        [],
        config,
    )
    assert decision.response_mode == "append"


def test_block_emits_clarification_questions():
    decision = compute_response_mode(
        [{"subject": "a", "key": "r", "value_a": "b", "value_b": "c", "severity": "severe"}],
        [{"question": "Which one is correct?", "severity": "severe", "raised_by": "x"}],
        ExtractionConfig(),
    )
    assert decision.response_mode == "block"
    assert "Which one is correct?" in decision.clarification_questions
