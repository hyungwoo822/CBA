"""Tests for ExtractionConfig and WorkspaceConfig default wiring."""

from brain_agent.config.schema import BrainAgentConfig, ExtractionConfig, WorkspaceConfig


def test_extraction_config_defaults():
    cfg = ExtractionConfig()
    assert cfg.triage_model == "auto"
    assert cfg.extract_model == "auto"
    assert cfg.max_retry == 1
    assert cfg.enable_severity_block is True
    assert cfg.promotion_threshold_n == 3
    assert cfg.expression_override_block is True


def test_workspace_config_defaults():
    cfg = WorkspaceConfig()
    assert cfg.default_decay_policy == "normal"
    assert cfg.vault_size_threshold_mb == 10
    assert cfg.vault_dir == "vault"


def test_brain_agent_config_wires_extraction_and_workspace():
    cfg = BrainAgentConfig()
    assert isinstance(cfg.extraction, ExtractionConfig)
    assert isinstance(cfg.workspace, WorkspaceConfig)


def test_extraction_config_from_dict_override():
    cfg = BrainAgentConfig.from_dict({
        "extraction": {
            "enable_severity_block": False,
            "expression_override_block": False,
        },
    })
    assert cfg.extraction.enable_severity_block is False
    assert cfg.extraction.expression_override_block is False
