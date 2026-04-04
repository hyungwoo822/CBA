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
