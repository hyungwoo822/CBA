"""Tests for Predictive Coding module (Friston 2005)."""
import pytest
from brain_agent.core.predictor import Predictor


@pytest.fixture
def predictor():
    return Predictor()


def test_no_prediction_initially(predictor):
    assert predictor.last_prediction is None


def test_compute_surprise_no_prediction(predictor):
    surprise = predictor.compute_surprise([0.1] * 10)
    assert surprise == 0.5


def test_compute_surprise_matching(predictor):
    predictor.store_prediction([0.5] * 10)
    surprise = predictor.compute_surprise([0.5] * 10)
    assert surprise < 0.1


def test_compute_surprise_mismatching(predictor):
    predictor.store_prediction([1.0] * 10)
    surprise = predictor.compute_surprise([-1.0] * 10)
    assert surprise > 0.7


def test_store_prediction(predictor):
    emb = [0.1, 0.2, 0.3]
    predictor.store_prediction(emb)
    assert predictor.last_prediction == emb


def test_surprise_range(predictor):
    predictor.store_prediction([1.0] * 10)
    surprise = predictor.compute_surprise([0.0] * 10)
    assert 0.0 <= surprise <= 1.0


def test_cosine_sim_identical():
    assert Predictor._cosine_sim([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    assert Predictor._cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_sim_empty():
    assert Predictor._cosine_sim([], [1, 0]) == 0.0
