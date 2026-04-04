from brain_agent.config.schema import BrainAgentConfig, MemoryConfig, RetrievalWeights


def test_default_config():
    cfg = BrainAgentConfig()
    assert cfg.agent.model == "openai/gpt-4o-mini"
    assert cfg.memory.working_capacity == 4
    assert cfg.dashboard.enabled is False


def test_memory_config_bounds():
    mc = MemoryConfig(working_capacity=4)
    assert mc.working_capacity == 4


def test_retrieval_weights_sum():
    rw = RetrievalWeights()
    total = rw.alpha + rw.beta + rw.gamma + rw.delta + rw.epsilon
    assert abs(total - 1.0) < 0.001


def test_from_dict():
    cfg = BrainAgentConfig.from_dict({"agent": {"model": "anthropic/claude-sonnet-4-20250514"}})
    assert cfg.agent.model == "anthropic/claude-sonnet-4-20250514"


def test_config_preserves_defaults():
    cfg = BrainAgentConfig.from_dict({})
    assert cfg.memory.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.agent.max_tool_iterations == 40
