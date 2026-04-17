"""ExtractionConfig per-stage model settings."""
from __future__ import annotations

from brain_agent.config.schema import ExtractionConfig


def test_extraction_config_defaults():
    config = ExtractionConfig()
    assert config.triage_model == "auto"
    assert config.extract_model == "auto"
    assert config.temporal_classify_model == "auto"
    assert config.refine_model == "auto"


def test_extraction_config_override():
    config = ExtractionConfig(extract_model="openai/gpt-4o-mini")
    assert config.extract_model == "openai/gpt-4o-mini"
    assert config.triage_model == "auto"
