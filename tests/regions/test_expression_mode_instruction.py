"""Expression-mode instruction wording tests."""

from brain_agent.regions.prefrontal import EXPRESSION_MODE_INSTRUCTION


def test_mentions_current_workspace():
    assert "CURRENT WORKSPACE" in EXPRESSION_MODE_INSTRUCTION


def test_mentions_contradiction_hint():
    assert "contradict" in EXPRESSION_MODE_INSTRUCTION.lower()


def test_mentions_open_questions_path():
    assert "기억에 없어" in EXPRESSION_MODE_INSTRUCTION or "모르겠어" in EXPRESSION_MODE_INSTRUCTION
