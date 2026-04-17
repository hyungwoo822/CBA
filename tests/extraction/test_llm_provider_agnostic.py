"""LLM provider agnosticism tests for extraction."""
import pytest

from brain_agent.providers.base import LLMProvider
from tests.extraction.test_orchestrator import _stub_memory as _stub_memory_from_orch


@pytest.fixture
def memory():
    return _stub_memory_from_orch()


async def test_recorder_is_an_llm_provider(mock_llm):
    assert isinstance(mock_llm, LLMProvider)


async def test_recorder_records_model_arg(mock_llm):
    mock_llm.enqueue_content('{"ok": true}')
    await mock_llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5",
    )
    assert mock_llm.calls[0]["model"] == "claude-haiku-4-5"


async def test_recorder_default_model_fallback(mock_llm):
    mock_llm.enqueue_content('{"ok": true}')
    await mock_llm.chat(messages=[{"role": "user", "content": "hi"}])
    assert mock_llm.get_default_model() == "mock-default"


async def test_recorder_runs_out_of_responses_raises(mock_llm):
    with pytest.raises(AssertionError, match="exhausted"):
        await mock_llm.chat(messages=[{"role": "user", "content": "hi"}])


async def test_recorder_canned_sequence(mock_llm):
    mock_llm.enqueue_content('{"first": 1}')
    mock_llm.enqueue_content('{"second": 2}')
    first = await mock_llm.chat(messages=[])
    second = await mock_llm.chat(messages=[])
    assert first.content == '{"first": 1}'
    assert second.content == '{"second": 2}'
    assert len(mock_llm.calls) == 2


async def test_all_stage_model_strings_honoured(memory, mock_llm):
    import json
    from unittest.mock import AsyncMock

    from brain_agent.extraction.config import ExtractionConfig
    from brain_agent.extraction.orchestrator import ExtractionOrchestrator

    config = ExtractionConfig(
        triage_model="triage-xx",
        extract_model="extract-xx",
        temporal_classify_model="tc-xx",
        refine_model="refine-xx",
    )
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "go", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "use",
                        "target": "go",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "use go",
            }
        )
    )
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "old-1",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
                "confidence": "EXTRACTED",
            }
        ]
    )
    mock_llm.enqueue_content("update")
    mock_llm.enqueue_content("polished")

    orchestrator = ExtractionOrchestrator(memory=memory, llm_provider=mock_llm, config=config)
    await orchestrator.extract(
        text="use go",
        session_id="s1",
        agent_response="draft",
        language="ko",
    )
    models_used = [call["model"] for call in mock_llm.calls]
    assert "extract-xx" in models_used
    assert "tc-xx" in models_used
    assert "refine-xx" in models_used


async def test_auto_falls_back_to_provider_default(memory, mock_llm):
    import json

    from brain_agent.extraction.config import ExtractionConfig
    from brain_agent.extraction.orchestrator import ExtractionOrchestrator

    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [],
                "edges": [],
                "new_type_proposals": [],
                "narrative_chunk": "x",
            }
        )
    )
    mock_llm.enqueue_content("polished")
    orchestrator = ExtractionOrchestrator(
        memory=memory,
        llm_provider=mock_llm,
        config=ExtractionConfig(),
    )
    await orchestrator.extract(
        text="random text",
        session_id="s1",
        agent_response="draft",
        language="ko",
    )
    assert [call["model"] for call in mock_llm.calls] == ["mock-default", "mock-default"]


async def test_no_anthropic_or_openai_imports_in_extraction_package():
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent.parent / "brain_agent" / "extraction"
    forbidden = re.compile(
        r"^\s*(?:import|from)\s+(anthropic|openai)(?:\s|$|\.)",
        re.MULTILINE,
    )
    offenders = []
    for py_file in root.rglob("*.py"):
        if forbidden.search(py_file.read_text(encoding="utf-8")):
            offenders.append(str(py_file))
    assert offenders == []
