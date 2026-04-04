import pytest
from brain_agent.agent import BrainAgent


@pytest.fixture
async def agent(tmp_path):
    a = BrainAgent(data_dir=str(tmp_path), use_mock_embeddings=True)
    await a.initialize()
    yield a
    await a.close()


async def test_basic_process(agent):
    result = await agent.process("hello world")
    assert result.response != ""
    assert result.signals_processed > 0


async def test_memory_encoded_after_process(agent):
    await agent.process("find the auth bug")
    stats = await agent.memory.stats()
    assert stats["staging"] >= 1


async def test_session_context_manager(tmp_path):
    agent = BrainAgent(data_dir=str(tmp_path), use_mock_embeddings=True)
    async with agent.session() as s:
        r1 = await s.send("first message")
        r2 = await s.send("second message")
        assert r1.response != ""
        assert r2.response != ""


async def test_async_context_manager(tmp_path):
    async with BrainAgent(data_dir=str(tmp_path), use_mock_embeddings=True) as agent:
        result = await agent.process("test")
        assert result.response != ""


async def test_config_override(tmp_path):
    agent = BrainAgent(
        data_dir=str(tmp_path),
        model="anthropic/claude-sonnet-4-20250514",
        use_mock_embeddings=True,
    )
    assert agent.config.agent.model == "anthropic/claude-sonnet-4-20250514"


async def test_memory_stats(agent):
    await agent.process("something")
    stats = await agent.memory.stats()
    assert "sensory" in stats
    assert "working" in stats
    assert "staging" in stats
    assert "episodic" in stats
