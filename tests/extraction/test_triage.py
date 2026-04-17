"""Tests for Stage 1 Triage."""
from unittest.mock import AsyncMock

import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.triage import Triage


@pytest.fixture
def fake_workspace_store():
    store = AsyncMock()
    store.get_session_workspace = AsyncMock(return_value="personal")

    async def _get_workspace(identifier):
        if identifier == "personal":
            return {"id": "personal", "name": "Personal Knowledge"}
        if identifier == "billing-service":
            return {"id": "billing-service", "name": "Billing Service"}
        return None

    store.get_workspace = _get_workspace
    store.list_workspaces = AsyncMock(
        return_value=[
            {"id": "personal", "name": "Personal Knowledge"},
            {"id": "billing-service", "name": "Billing Service"},
        ]
    )
    return store


@pytest.fixture
def triage(fake_workspace_store, mock_llm):
    return Triage(fake_workspace_store, mock_llm, ExtractionConfig())


async def test_greeting_only_skips_stages_2_and_3(triage):
    result = await triage.classify("hello hyungpu", "s1", None)
    assert "greeting" in result.input_kinds
    assert 2 in result.skip_stages
    assert 3 in result.skip_stages
    assert result.target_workspace_id == "personal"


async def test_spec_drop_runs_full_pipeline(triage):
    result = await triage.classify("billing flow retry policy is 3 attempts", "s1", None)
    assert "spec_drop" in result.input_kinds
    assert result.skip_stages == []


async def test_multi_label_confirmation_plus_correction_runs_full(triage):
    result = await triage.classify(
        "yes, actually change the idempotency key to a header",
        "s1",
        None,
    )
    assert "confirmation" in result.input_kinds
    assert "correction" in result.input_kinds
    assert result.skip_stages == []


async def test_empty_patterns_default_to_spec_drop(triage):
    result = await triage.classify(
        "flux capacitor operates at one point twenty one gigawatts",
        "s1",
        None,
    )
    assert result.input_kinds == ["spec_drop"]


async def test_question_partial_skip(triage):
    result = await triage.classify("what should I eat?", "s1", None)
    assert "question" in result.input_kinds
    assert 2 in result.skip_stages
    assert 3 not in result.skip_stages


async def test_workspace_override_emits_ask(triage):
    result = await triage.classify(
        "billing-service retry policy changes to 3",
        "s1",
        {"workspace_hint": "billing-service", "confidence": 0.9},
    )
    assert result.workspace_ask is not None
    assert "billing service" in result.workspace_ask.lower()


async def test_workspace_override_without_hint_stays_in_current(triage):
    result = await triage.classify("billing-service is fine", "s1", None)
    assert result.target_workspace_id == "personal"
    assert result.workspace_ask is None


async def test_session_workspace_falls_back_to_personal(fake_workspace_store, mock_llm):
    fake_workspace_store.get_session_workspace = AsyncMock(return_value=None)
    triage = Triage(fake_workspace_store, mock_llm, ExtractionConfig())
    result = await triage.classify("hi", "unknown-session", None)
    assert result.target_workspace_id == "personal"
