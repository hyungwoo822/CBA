"""Broca block-mode formatting tests."""

import pytest

from brain_agent.regions.broca import BrocaArea


@pytest.mark.asyncio
async def test_format_response_block_mode_renders_questions():
    b = BrocaArea(llm_provider=None)
    text = await b.format_response(
        pfc_output=None,
        response_mode="block",
        clarification_questions=["집이 바뀌었어?", "언제 이사했어?"],
        language="ko",
    )
    assert "집이 바뀌었어?" in text
    assert "언제 이사했어?" in text


@pytest.mark.asyncio
async def test_format_response_block_mode_english():
    b = BrocaArea(llm_provider=None)
    text = await b.format_response(
        pfc_output=None,
        response_mode="block",
        clarification_questions=["Did you move?", "When?"],
        language="en",
    )
    assert "Did you move?" in text
    assert "When?" in text


@pytest.mark.asyncio
async def test_format_response_normal_mode_returns_pfc_output():
    b = BrocaArea(llm_provider=None)
    text = await b.format_response(
        pfc_output="Hello there, how are you?",
        response_mode="normal",
        clarification_questions=[],
        language="en",
    )
    assert text == "Hello there, how are you?"


@pytest.mark.asyncio
async def test_format_response_block_mode_with_no_questions_falls_back_to_pfc():
    b = BrocaArea(llm_provider=None)
    text = await b.format_response(
        pfc_output="fallback answer",
        response_mode="block",
        clarification_questions=[],
        language="en",
    )
    assert text == "fallback answer"
