"""Wernicke workspace_hint coverage."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def wernicke_with_llm():
    from brain_agent.regions.wernicke import WernickeArea

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock()
    return WernickeArea(llm_provider=mock_llm), mock_llm


@pytest.mark.asyncio
async def test_workspace_hint_present_when_llm_returns_it(wernicke_with_llm):
    w, llm = wernicke_with_llm
    llm.chat.return_value = MagicMock(content=(
        '{"intent":"question","complexity":"simple","keywords":["payments"],'
        '"semantic_roles":{},"discourse_type":"technical","language":"en",'
        '"workspace_hint":"billing-service","workspace_hint_confidence":0.85}'
    ))

    result = await w._comprehend_with_llm("what endpoint handles /payments?")

    assert result["workspace_hint"] == "billing-service"
    assert result["workspace_hint_confidence"] == 0.85


@pytest.mark.asyncio
async def test_workspace_hint_absent_when_llm_omits(wernicke_with_llm):
    w, llm = wernicke_with_llm
    llm.chat.return_value = MagicMock(content=(
        '{"intent":"greeting","complexity":"simple","keywords":["hi"],'
        '"semantic_roles":{},"discourse_type":"social","language":"en"}'
    ))

    result = await w._comprehend_with_llm("hi")

    assert "workspace_hint" not in result or result["workspace_hint"] is None


def test_m1_annotation_present_in_source():
    import inspect
    import brain_agent.regions.wernicke as wmod

    src = inspect.getsource(wmod)
    assert "M1" in src
    assert "pragmatic" in src.lower() or "TPJ" in src or "dlPFC" in src
