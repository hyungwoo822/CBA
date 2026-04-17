import pytest
import numpy as np


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_brain.db")


@pytest.fixture
def mock_embedding():
    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    return _embed


@pytest.fixture
async def memory_manager(tmp_path, mock_embedding):
    """MemoryManager with personal workspace and universal ontology seed."""
    from brain_agent.memory.manager import MemoryManager

    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
async def personal_workspace_id(memory_manager):
    workspace = await memory_manager.workspace.get_or_create_personal()
    return workspace["id"]
