"""Phase 2 end-to-end smoke tests through MemoryManager."""


async def test_contradictions_batch_retrieval_pattern(
    memory_manager, personal_workspace_id
):
    ws = personal_workspace_id
    c1 = await memory_manager.contradictions.detect(
        workspace_id=ws,
        subject="Alice",
        key_or_relation="role",
        value_a="manager",
        value_b="engineer",
        value_a_confidence="EXTRACTED",
        value_b_confidence="EXTRACTED",
    )
    c2 = await memory_manager.contradictions.detect(
        workspace_id=ws,
        subject="Alice",
        key_or_relation="team",
        value_a="payments",
        value_b="billing",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    c3 = await memory_manager.contradictions.detect(
        workspace_id=ws,
        subject="Bob",
        key_or_relation="role",
        value_a="ic",
        value_b="lead",
        value_a_confidence="EXTRACTED",
        value_b_confidence="INFERRED",
    )
    assert c1["severity"] == "severe"
    assert c2["severity"] == "minor"
    assert c3["severity"] == "moderate"

    batch = await memory_manager.contradictions.get_for_subject_batch(
        ws, ["Alice", "Bob", "Carol"]
    )
    assert len(batch["Alice"]) == 2
    assert len(batch["Bob"]) == 1
    assert batch["Carol"] == []

    await memory_manager.contradictions.resolve(
        c2["id"], resolution="confirmed billing team", resolved_by="user"
    )
    batch_after = await memory_manager.contradictions.get_for_subject_batch(
        ws, ["Alice"]
    )
    assert len(batch_after["Alice"]) == 1
    assert batch_after["Alice"][0]["id"] == c1["id"]


async def test_open_questions_severe_blocking_lifecycle(
    memory_manager, personal_workspace_id
):
    ws = personal_workspace_id
    q_severe = await memory_manager.open_questions.add_question(
        workspace_id=ws,
        question="Which billing policy applies?",
        raised_by="contradiction",
        severity="severe",
        context_node="BillingPolicy",
    )
    q_minor = await memory_manager.open_questions.add_question(
        workspace_id=ws,
        question="clarify typo?",
        raised_by="ambiguity_detector",
        severity="minor",
    )
    assert q_severe["blocking"] == 1
    assert q_minor["blocking"] == 0

    blocking = await memory_manager.open_questions.list_blocking(ws)
    assert [item["id"] for item in blocking] == [q_severe["id"]]
    assert await memory_manager.open_questions.count_blocking(ws) == 1

    await memory_manager.open_questions.answer_question(
        q_severe["id"], answer="policy v2 applies", answer_source="src-1"
    )
    assert await memory_manager.open_questions.count_blocking(ws) == 0
    unanswered = await memory_manager.open_questions.list_unanswered(ws)
    assert [item["id"] for item in unanswered] == [q_minor["id"]]
