"""Tests for ContradictionsStore CRUD, severity, and retrieval helpers."""
import pytest

from brain_agent.memory.contradictions_store import (
    ContradictionsStore,
    _compute_severity,
)


@pytest.mark.parametrize(
    "conf_a, conf_b, subject, core_set, relation, expected",
    [
        ("EXTRACTED", "EXTRACTED", "node_x", frozenset(), "prefers", "severe"),
        (
            "INFERRED",
            "INFERRED",
            "hub_node",
            frozenset({"hub_node"}),
            "prefers",
            "severe",
        ),
        ("INFERRED", "INFERRED", "node_x", frozenset(), "supersedes", "severe"),
        ("INFERRED", "INFERRED", "node_x", frozenset(), "contradicts", "severe"),
        (
            "EXTRACTED",
            "INFERRED",
            "hub_node",
            frozenset({"hub_node"}),
            "prefers",
            "severe",
        ),
        ("INFERRED", "INFERRED", "node_x", frozenset(), "prefers", "minor"),
        ("INFERRED_LOW", "INFERRED", "node_x", frozenset(), "prefers", "minor"),
        ("EXTRACTED", "INFERRED", "node_x", frozenset(), "prefers", "moderate"),
        ("EXTRACTED", "", "node_x", frozenset(), "prefers", "moderate"),
        ("UNKNOWN", "OTHER", "node_x", frozenset(), "prefers", "moderate"),
    ],
)
def test_compute_severity_matrix(conf_a, conf_b, subject, core_set, relation, expected):
    result = _compute_severity(
        value_a_confidence=conf_a,
        value_b_confidence=conf_b,
        subject_node=subject,
        core_node_set=core_set,
        key_or_relation=relation,
    )
    assert result == expected


def test_compute_severity_default_core_set_is_empty():
    result = _compute_severity(
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
        subject_node="anything",
        core_node_set=None,
        key_or_relation="prefers",
    )
    assert result == "minor"


def test_compute_severity_severe_precedence_over_minor():
    result = _compute_severity(
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
        subject_node="hub",
        core_node_set=frozenset({"hub"}),
        key_or_relation="prefers",
    )
    assert result == "severe"


@pytest.fixture
async def store(tmp_path):
    s = ContradictionsStore(db_path=str(tmp_path / "contradictions.db"))
    await s.initialize()
    yield s
    await s.close()


async def test_detect_creates_row_with_auto_severity(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_confidence="EXTRACTED",
        value_b_confidence="EXTRACTED",
    )
    assert row["id"]
    assert row["workspace_id"] == "ws_a"
    assert row["subject_node"] == "Alice"
    assert row["key_or_relation"] == "prefers_coffee"
    assert row["value_a"] == "yes"
    assert row["value_b"] == "no"
    assert row["severity"] == "severe"
    assert row["status"] == "open"
    assert row["detected_at"]
    assert row["resolved_at"] is None


async def test_detect_both_inferred_is_minor(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    assert row["severity"] == "minor"


async def test_detect_mixed_confidence_is_moderate(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_confidence="EXTRACTED",
        value_b_confidence="INFERRED",
    )
    assert row["severity"] == "moderate"


async def test_detect_hub_subject_is_severe(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="HubNode",
        key_or_relation="prefers",
        value_a="yes",
        value_b="no",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
        core_node_set={"HubNode"},
    )
    assert row["severity"] == "severe"


async def test_detect_supersedes_relation_is_severe(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="v1",
        key_or_relation="supersedes",
        value_a="v2",
        value_b="v3",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    assert row["severity"] == "severe"


async def test_detect_with_source_and_confidence_strings_roundtrip(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_source="source-1",
        value_b_source="source-2",
        value_a_confidence="EXTRACTED",
        value_b_confidence="INFERRED",
    )
    assert row["value_a_source"] == "source-1"
    assert row["value_b_source"] == "source-2"
    assert row["value_a_confidence"] == "EXTRACTED"
    assert row["value_b_confidence"] == "INFERRED"


async def test_resolve_sets_status_and_fields(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.resolve(
        row["id"],
        resolution="prefers coffee (user confirmed)",
        resolved_by="user",
        resolution_confidence="EXTRACTED",
    )
    fresh = await store._get_by_id(row["id"])
    assert fresh["status"] == "resolved"
    assert fresh["resolution"] == "prefers coffee (user confirmed)"
    assert fresh["resolved_by"] == "user"
    assert fresh["resolution_confidence"] == "EXTRACTED"
    assert fresh["resolved_at"] is not None


async def test_resolve_unknown_id_raises(store):
    with pytest.raises(ValueError, match="not found"):
        await store.resolve("no-such-id", resolution="x")


async def test_dismiss_sets_status_dismissed(store):
    row = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="prefers",
        value_a="a",
        value_b="b",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.dismiss(row["id"])
    fresh = await store._get_by_id(row["id"])
    assert fresh["status"] == "dismissed"
    assert fresh["resolved_at"] is not None


async def test_list_open_excludes_resolved_and_dismissed(store):
    c1 = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    c2 = await store.detect(
        workspace_id="ws_a",
        subject="Bob",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    c3 = await store.detect(
        workspace_id="ws_a",
        subject="Carol",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.resolve(c1["id"], resolution="r")
    await store.dismiss(c2["id"])

    openings = await store.list_open("ws_a")
    ids = {row["id"] for row in openings}
    assert ids == {c3["id"]}


async def test_list_open_scoped_by_workspace(store):
    await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.detect(
        workspace_id="ws_b",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    assert len(await store.list_open("ws_a")) == 1
    assert len(await store.list_open("ws_b")) == 1


async def test_list_by_severity(store):
    await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="EXTRACTED",
        value_b_confidence="EXTRACTED",
    )
    await store.detect(
        workspace_id="ws_a",
        subject="Bob",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    severes = await store.list_by_severity("ws_a", "severe")
    minors = await store.list_by_severity("ws_a", "minor")
    assert len(severes) == 1
    assert len(minors) == 1
    assert severes[0]["subject_node"] == "Alice"
    assert minors[0]["subject_node"] == "Bob"


async def test_get_for_subject_returns_only_open(store):
    c_open = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    c_resolved = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k2",
        value_a="p",
        value_b="q",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.resolve(c_resolved["id"], resolution="done")

    results = await store.get_for_subject("ws_a", "Alice")
    assert len(results) == 1
    assert results[0]["id"] == c_open["id"]


async def test_get_for_subject_empty_when_no_match(store):
    assert await store.get_for_subject("ws_a", "NonExistent") == []


async def test_get_for_subject_scoped_by_workspace(store):
    await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    assert len(await store.get_for_subject("ws_b", "Alice")) == 0


async def test_get_for_subject_batch_multiple_subjects(store):
    await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k2",
        value_a="p",
        value_b="q",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.detect(
        workspace_id="ws_a",
        subject="Bob",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )

    out = await store.get_for_subject_batch("ws_a", ["Alice", "Bob", "Carol"])
    assert set(out.keys()) == {"Alice", "Bob", "Carol"}
    assert len(out["Alice"]) == 2
    assert len(out["Bob"]) == 1
    assert out["Carol"] == []


async def test_get_for_subject_batch_empty_subject_list(store):
    out = await store.get_for_subject_batch("ws_a", [])
    assert out == {}


async def test_get_for_subject_batch_excludes_resolved(store):
    c1 = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k",
        value_a="x",
        value_b="y",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    c2 = await store.detect(
        workspace_id="ws_a",
        subject="Alice",
        key_or_relation="k2",
        value_a="p",
        value_b="q",
        value_a_confidence="INFERRED",
        value_b_confidence="INFERRED",
    )
    await store.resolve(c1["id"], resolution="x")
    out = await store.get_for_subject_batch("ws_a", ["Alice"])
    assert len(out["Alice"]) == 1
    assert out["Alice"][0]["id"] == c2["id"]
