"""Tests for OpenQuestionsStore question queue and blocking auto-assign."""
import pytest

from brain_agent.memory.open_questions_store import OpenQuestionsStore


@pytest.fixture
async def store(tmp_path):
    s = OpenQuestionsStore(db_path=str(tmp_path / "open_questions.db"))
    await s.initialize()
    yield s
    await s.close()


async def test_add_question_default_moderate_is_nonblocking(store):
    q = await store.add_question(
        workspace_id="ws_a",
        question="Is Alice a manager or engineer?",
        raised_by="ambiguity_detector",
    )
    assert q["id"]
    assert q["workspace_id"] == "ws_a"
    assert q["question"] == "Is Alice a manager or engineer?"
    assert q["raised_by"] == "ambiguity_detector"
    assert q["severity"] == "moderate"
    assert q["blocking"] == 0
    assert q["asked_at"]
    assert q["answered_at"] is None


async def test_add_question_severe_auto_sets_blocking(store):
    q = await store.add_question(
        workspace_id="ws_a",
        question="Are these two SLAs contradictory?",
        raised_by="contradiction",
        severity="severe",
    )
    assert q["severity"] == "severe"
    assert q["blocking"] == 1


async def test_add_question_minor_is_nonblocking(store):
    q = await store.add_question(
        workspace_id="ws_a",
        question="minor clarification",
        raised_by="unknown_fact",
        severity="minor",
    )
    assert q["severity"] == "minor"
    assert q["blocking"] == 0


async def test_add_question_stores_context(store):
    q = await store.add_question(
        workspace_id="ws_a",
        question="Which billing API?",
        raised_by="user",
        context_node="BillingService",
        context_input="we need to invoice customer X",
    )
    assert q["context_node"] == "BillingService"
    assert q["context_input"] == "we need to invoice customer X"


async def test_add_question_rejects_unknown_raised_by(store):
    with pytest.raises(Exception):
        await store.add_question(
            workspace_id="ws_a",
            question="x",
            raised_by="not_a_valid_source",
        )


async def test_add_question_rejects_unknown_severity(store):
    with pytest.raises(Exception):
        await store.add_question(
            workspace_id="ws_a",
            question="x",
            raised_by="user",
            severity="catastrophic",
        )


async def test_answer_question_sets_answer_and_timestamp(store):
    q = await store.add_question(
        workspace_id="ws_a",
        question="Which region?",
        raised_by="unknown_fact",
    )
    await store.answer_question(q["id"], answer="us-west-2", answer_source="source-1")
    fresh = await store._get_by_id(q["id"])
    assert fresh["answer"] == "us-west-2"
    assert fresh["answer_source"] == "source-1"
    assert fresh["answered_at"] is not None


async def test_answer_unknown_id_raises(store):
    with pytest.raises(ValueError, match="not found"):
        await store.answer_question("no-such-id", answer="x")


async def test_list_unanswered_excludes_answered(store):
    q1 = await store.add_question(
        workspace_id="ws_a",
        question="Q1",
        raised_by="user",
    )
    q2 = await store.add_question(
        workspace_id="ws_a",
        question="Q2",
        raised_by="user",
    )
    await store.answer_question(q1["id"], answer="a1")
    out = await store.list_unanswered("ws_a")
    ids = {q["id"] for q in out}
    assert ids == {q2["id"]}


async def test_list_unanswered_scoped_by_workspace(store):
    await store.add_question(workspace_id="ws_a", question="Q", raised_by="user")
    await store.add_question(workspace_id="ws_b", question="Q", raised_by="user")
    assert len(await store.list_unanswered("ws_a")) == 1
    assert len(await store.list_unanswered("ws_b")) == 1


async def test_list_blocking_returns_only_blocking_and_unanswered(store):
    q_severe = await store.add_question(
        workspace_id="ws_a",
        question="Severe",
        raised_by="contradiction",
        severity="severe",
    )
    await store.add_question(
        workspace_id="ws_a",
        question="Moderate",
        raised_by="user",
        severity="moderate",
    )
    q_severe_answered = await store.add_question(
        workspace_id="ws_a",
        question="SevereDone",
        raised_by="contradiction",
        severity="severe",
    )
    await store.answer_question(q_severe_answered["id"], answer="done")

    out = await store.list_blocking("ws_a")
    ids = {q["id"] for q in out}
    assert ids == {q_severe["id"]}


async def test_list_by_severity(store):
    await store.add_question(
        workspace_id="ws_a",
        question="M1",
        raised_by="user",
        severity="minor",
    )
    await store.add_question(
        workspace_id="ws_a",
        question="M2",
        raised_by="user",
        severity="moderate",
    )
    await store.add_question(
        workspace_id="ws_a",
        question="S1",
        raised_by="user",
        severity="severe",
    )
    minors = await store.list_by_severity("ws_a", "minor")
    moderates = await store.list_by_severity("ws_a", "moderate")
    severes = await store.list_by_severity("ws_a", "severe")
    assert [q["question"] for q in minors] == ["M1"]
    assert [q["question"] for q in moderates] == ["M2"]
    assert [q["question"] for q in severes] == ["S1"]


async def test_count_blocking(store):
    await store.add_question(
        workspace_id="ws_a",
        question="S1",
        raised_by="contradiction",
        severity="severe",
    )
    await store.add_question(
        workspace_id="ws_a",
        question="S2",
        raised_by="contradiction",
        severity="severe",
    )
    await store.add_question(
        workspace_id="ws_a",
        question="M1",
        raised_by="user",
        severity="moderate",
    )
    q_done = await store.add_question(
        workspace_id="ws_a",
        question="S3",
        raised_by="contradiction",
        severity="severe",
    )
    await store.answer_question(q_done["id"], answer="x")

    assert await store.count_blocking("ws_a") == 2
    assert await store.count_blocking("ws_b") == 0


async def test_count_blocking_empty(store):
    assert await store.count_blocking("ws_a") == 0
