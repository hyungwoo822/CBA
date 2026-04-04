import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from brain_agent.regions.visual_cortex import VisualCortex
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def visual_cortex():
    return VisualCortex()


def test_anatomy(visual_cortex):
    assert visual_cortex.lobe == Lobe.OCCIPITAL
    assert visual_cortex.hemisphere == Hemisphere.BILATERAL
    assert visual_cortex.position.x == 0
    assert visual_cortex.position.y == -40
    assert visual_cortex.position.z == -10


async def test_processes_image_data(visual_cortex):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"\x89PNG\r\n"},
    )
    result = await visual_cortex.process(sig)
    assert result is not None
    assert result.payload["modality"] == "visual"
    assert result.payload["visual_features"]["size_bytes"] == 6


async def test_basic_features_description(visual_cortex):
    """Without llm_provider, basic features include a description."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"data"},
    )
    result = await visual_cortex.process(sig)
    assert "description" in result.payload["visual_features"]
    assert "4 bytes" in result.payload["visual_features"]["description"]


async def test_passthrough_no_image(visual_cortex):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    result = await visual_cortex.process(sig)
    assert result is sig
    assert "modality" not in result.payload or result.payload.get("modality") != "visual"


async def test_emits_activation_without_provider(visual_cortex):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"data"},
    )
    await visual_cortex.process(sig)
    assert visual_cortex.activation_level == pytest.approx(0.5)


async def test_no_text_override_when_text_present():
    """If payload already has 'text', do not overwrite it."""
    vc = VisualCortex()
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"data", "text": "keep me"},
    )
    result = await vc.process(sig)
    assert result.payload["text"] == "keep me"


async def test_text_set_from_description_when_missing():
    """If payload has no 'text', description fills it."""
    vc = VisualCortex()
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"data"},
    )
    result = await vc.process(sig)
    assert result.payload["text"] == result.payload["visual_features"]["description"]


async def test_vision_llm_called_when_provider_present():
    """When llm_provider is set, it should call the vision LLM."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "description": "A cat sitting on a table",
        "objects": ["cat", "table"],
        "text_content": "",
        "spatial_layout": "cat centered on table",
        "emotional_tone": "neutral",
    })
    mock_provider.chat = AsyncMock(return_value=mock_response)

    vc = VisualCortex(llm_provider=mock_provider)
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"\x89PNG\r\n"},
    )
    result = await vc.process(sig)

    mock_provider.chat.assert_called_once()
    assert result.payload["visual_features"]["description"] == "A cat sitting on a table"
    assert result.payload["visual_features"]["objects"] == ["cat", "table"]
    assert vc.activation_level == pytest.approx(0.8)


async def test_vision_llm_fallback_on_error():
    """If LLM call fails, fall back to basic features."""
    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(side_effect=Exception("API error"))

    vc = VisualCortex(llm_provider=mock_provider)
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"data"},
    )
    result = await vc.process(sig)
    assert result.payload["visual_features"]["size_bytes"] == 4


async def test_vision_llm_strips_markdown_fences():
    """LLM response wrapped in ```json fences should be parsed correctly."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '```json\n{"description": "test", "objects": []}\n```'
    mock_provider.chat = AsyncMock(return_value=mock_response)

    vc = VisualCortex(llm_provider=mock_provider)
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"img"},
    )
    result = await vc.process(sig)
    assert result.payload["visual_features"]["description"] == "test"


async def test_string_image_data_used_as_url():
    """String image_data (URL) passed directly to LLM."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({"description": "photo", "objects": []})
    mock_provider.chat = AsyncMock(return_value=mock_response)

    vc = VisualCortex(llm_provider=mock_provider)
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": "http://example.com/img.png"},
    )
    result = await vc.process(sig)

    # Check that the URL was passed as-is (not base64 encoded)
    call_args = mock_provider.chat.call_args
    messages = call_args[0][0]
    user_content = messages[1]["content"]
    image_part = [p for p in user_content if p["type"] == "image_url"][0]
    assert image_part["image_url"]["url"] == "http://example.com/img.png"


async def test_vision_model_override():
    """Custom vision_model is passed to provider."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({"description": "x", "objects": []})
    mock_provider.chat = AsyncMock(return_value=mock_response)

    vc = VisualCortex(llm_provider=mock_provider, vision_model="gpt-4o")
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"image_data": b"img"},
    )
    await vc.process(sig)

    call_kwargs = mock_provider.chat.call_args
    assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs[1].get("model") == "gpt-4o"
