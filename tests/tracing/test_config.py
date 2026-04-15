"""Tests for TracingConfig integration into BrainAgentConfig."""
from brain_agent.config.schema import BrainAgentConfig, TracingConfig


def test_tracing_config_defaults():
    config = TracingConfig()
    assert config.enabled is False
    assert config.project_name == "brain-agent"
    assert config.api_key == ""


def test_brain_agent_config_has_tracing():
    config = BrainAgentConfig()
    assert isinstance(config.tracing, TracingConfig)
    assert config.tracing.enabled is False


def test_tracing_config_from_dict():
    config = BrainAgentConfig.from_dict({
        "tracing": {"enabled": True, "project_name": "my-project", "api_key": "ls_test"}
    })
    assert config.tracing.enabled is True
    assert config.tracing.project_name == "my-project"
    assert config.tracing.api_key == "ls_test"
