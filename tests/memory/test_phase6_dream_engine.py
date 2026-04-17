"""Tests for Phase 6 DreamingEngine cross-workspace behavior."""

from brain_agent.memory.dreaming import DreamingEngine, RecallEntry, RecallTracker


def _record(tracker, *, key, content, source, workspace_id, count=5):
    entry = RecallEntry(key=key, content_preview=content, source=source)
    for i in range(count):
        entry.record_recall(query=f"q{i}", score=0.95)
    entry.concept_tags = []
    entry.origin_workspace_id = workspace_id
    tracker._entries[key] = entry
    tracker._loaded = True


async def test_dream_cycle_spans_all_workspaces(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._DREAMS_DIR",
        str(tmp_path / "dreams"),
    )
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._RECALL_STORE_PATH",
        str(tmp_path / "dreams" / "recall.json"),
    )
    tracker = RecallTracker()
    _record(
        tracker,
        key="memory:a",
        content="biz fact",
        source="memory",
        workspace_id="biz",
    )
    _record(
        tracker,
        key="memory:b",
        content="personal fact",
        source="memory",
        workspace_id="personal",
    )
    engine = DreamingEngine(tracker, mode="core")
    promoted = await engine.run_cycle()
    keys = {c["key"] for c in promoted}
    assert "memory:a" in keys
    assert "memory:b" in keys


async def test_dream_cycle_preserves_origin_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._DREAMS_DIR",
        str(tmp_path / "dreams"),
    )
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._RECALL_STORE_PATH",
        str(tmp_path / "dreams" / "recall.json"),
    )
    tracker = RecallTracker()
    _record(
        tracker,
        key="memory:a",
        content="biz fact",
        source="memory",
        workspace_id="biz",
    )
    engine = DreamingEngine(tracker, mode="core")
    promoted = await engine.run_cycle()
    assert promoted[0]["entry"].origin_workspace_id == "biz"


async def test_recall_entry_origin_workspace_round_trips(tmp_path):
    entry = RecallEntry(key="memory:x", content_preview="foo", source="memory")
    entry.origin_workspace_id = "biz"
    entry.record_recall("q", 0.9)
    data = entry.to_dict()
    assert data["origin_workspace_id"] == "biz"
    restored = RecallEntry.from_dict(data)
    assert restored.origin_workspace_id == "biz"


async def test_recall_tracker_record_accepts_workspace_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._DREAMS_DIR",
        str(tmp_path / "dreams"),
    )
    monkeypatch.setattr(
        "brain_agent.memory.dreaming._RECALL_STORE_PATH",
        str(tmp_path / "dreams" / "recall.json"),
    )
    tracker = RecallTracker()
    tracker.record(
        memory_id="x",
        content="c",
        query="q",
        score=0.9,
        source="memory",
        workspace_id="biz",
    )
    entries = tracker.get_entries()
    assert list(entries.values())[0].origin_workspace_id == "biz"
