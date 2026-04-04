from brain_agent.memory.working_memory import WorkingMemory, WorkingMemoryItem, SLOT_CAPACITY


# ── Existing tests (backward compatibility) ──────────────────────────

def test_load_within_capacity():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    assert len(wm.get_slots()) == 2

def test_capacity_limit_evicts_oldest():
    wm = WorkingMemory(capacity=4)
    for i in range(6):
        wm.load(WorkingMemoryItem(content=f"item_{i}", slot="phonological"))
    slots = wm.get_slots()
    assert len(slots) == 4
    contents = [s.content for s in slots]
    assert "item_0" not in contents
    assert "item_1" not in contents
    assert "item_5" in contents

def test_rehearse_prevents_eviction():
    wm = WorkingMemory(capacity=3)
    wm.load(WorkingMemoryItem(content="important", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    wm.load(WorkingMemoryItem(content="c", slot="phonological"))
    wm.rehearse("important")
    wm.load(WorkingMemoryItem(content="d", slot="phonological"))
    contents = [s.content for s in wm.get_slots()]
    assert "important" in contents
    assert "b" not in contents

def test_evict_returns_evicted_items():
    wm = WorkingMemory(capacity=2)
    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    evicted = wm.load(WorkingMemoryItem(content="c", slot="phonological"))
    assert len(evicted) == 1
    assert evicted[0].content == "a"

def test_session_boundary_clears_irrelevant():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="auth bug", slot="episodic"))
    wm.load(WorkingMemoryItem(content="weather", slot="episodic"))
    wm.on_session_boundary(lambda item: "auth" in item.content)
    assert len(wm.get_slots()) == 1
    assert wm.get_slots()[0].content == "auth bug"

def test_get_context_returns_all_contents():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="fact 1", slot="phonological"))
    wm.load(WorkingMemoryItem(content="fact 2", slot="episodic"))
    ctx = wm.get_context()
    assert "fact 1" in ctx
    assert "fact 2" in ctx


# ── Multi-component capacity enforcement (Baddeley 2000) ─────────────

def test_phonological_capacity_independent():
    """Phonological component enforces its own capacity limit."""
    wm = WorkingMemory(capacity=4)
    for i in range(6):
        wm.load(WorkingMemoryItem(content=f"word_{i}", slot="phonological"))
    phon = wm.get_component("phonological")
    assert len(phon) == 4
    assert phon[0].content == "word_2"
    assert phon[-1].content == "word_5"


def test_visuospatial_capacity_independent():
    """Visuospatial component enforces its own capacity limit (3 items)."""
    wm = WorkingMemory(capacity=4)
    for i in range(5):
        wm.load(WorkingMemoryItem(content=f"spatial_{i}", slot="visuospatial"))
    vis = wm.get_component("visuospatial")
    assert len(vis) == SLOT_CAPACITY["visuospatial"]  # 3
    assert vis[0].content == "spatial_2"
    assert vis[-1].content == "spatial_4"


def test_episodic_buffer_capacity_independent():
    """Episodic buffer enforces its own capacity limit (4 chunks)."""
    wm = WorkingMemory(capacity=4)
    for i in range(6):
        wm.load(WorkingMemoryItem(content=f"chunk_{i}", slot="episodic_buffer"))
    eb = wm.get_component("episodic_buffer")
    assert len(eb) == SLOT_CAPACITY["episodic_buffer"]  # 4
    assert eb[0].content == "chunk_2"
    assert eb[-1].content == "chunk_5"


def test_components_do_not_share_capacity():
    """Filling one component does not evict from another."""
    wm = WorkingMemory(capacity=4)
    # Fill phonological to capacity
    for i in range(4):
        wm.load(WorkingMemoryItem(content=f"word_{i}", slot="phonological"))
    # Fill visuospatial to capacity
    for i in range(3):
        wm.load(WorkingMemoryItem(content=f"spatial_{i}", slot="visuospatial"))
    # Fill episodic buffer to capacity
    for i in range(4):
        wm.load(WorkingMemoryItem(content=f"chunk_{i}", slot="episodic_buffer"))

    # All components full, none evicted across boundaries
    assert len(wm.get_component("phonological")) == 4
    assert len(wm.get_component("visuospatial")) == 3
    assert len(wm.get_component("episodic_buffer")) == 4
    assert len(wm.get_slots()) == 11  # 4 + 3 + 4


def test_eviction_within_component_only():
    """Eviction only affects the target component, not others."""
    wm = WorkingMemory(capacity=2)
    wm.load(WorkingMemoryItem(content="word_0", slot="phonological"))
    wm.load(WorkingMemoryItem(content="word_1", slot="phonological"))
    wm.load(WorkingMemoryItem(content="spatial_0", slot="visuospatial"))

    # Evict from phonological by adding a 3rd
    evicted = wm.load(WorkingMemoryItem(content="word_2", slot="phonological"))
    assert len(evicted) == 1
    assert evicted[0].content == "word_0"
    # Visuospatial untouched
    assert len(wm.get_component("visuospatial")) == 1
    assert wm.get_component("visuospatial")[0].content == "spatial_0"


def test_unknown_slot_routes_to_phonological():
    """Items with unrecognized slot names route to phonological."""
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="mystery", slot="unknown_slot"))
    assert len(wm.get_component("phonological")) == 1
    assert wm.get_component("phonological")[0].content == "mystery"


def test_get_component_unknown_returns_empty():
    """Requesting a non-existent component returns an empty list."""
    wm = WorkingMemory()
    assert wm.get_component("nonexistent") == []


# ── Episodic buffer integration ──────────────────────────────────────

def test_bind_to_episodic_buffer_loads_memories():
    """bind_to_episodic_buffer loads retrieved LTM fragments into the buffer."""
    wm = WorkingMemory()
    retrieved = [
        {"id": "mem1", "content": "Python is a language"},
        {"id": "mem2", "content": "Django is a framework"},
    ]
    wm.bind_to_episodic_buffer(retrieved)
    eb = wm.get_component("episodic_buffer")
    assert len(eb) == 2
    assert eb[0].content == "Python is a language"
    assert eb[1].content == "Django is a framework"
    # Check linked_memories populated
    assert "mem1" in eb[0].linked_memories
    assert "mem2" in eb[1].linked_memories


def test_bind_to_episodic_buffer_respects_capacity():
    """bind_to_episodic_buffer caps at episodic_buffer capacity."""
    wm = WorkingMemory()
    # Send 8 memories, only 4 should fit
    retrieved = [{"id": f"m{i}", "content": f"memory {i}"} for i in range(8)]
    wm.bind_to_episodic_buffer(retrieved)
    eb = wm.get_component("episodic_buffer")
    assert len(eb) == SLOT_CAPACITY["episodic_buffer"]  # 4


def test_bind_to_episodic_buffer_evicts_oldest():
    """New LTM fragments evict oldest episodic buffer items."""
    wm = WorkingMemory()
    # Pre-fill buffer
    for i in range(4):
        wm.load(WorkingMemoryItem(content=f"old_{i}", slot="episodic_buffer"))
    # Now bind new memories — should evict old ones
    retrieved = [
        {"id": "new1", "content": "fresh memory 1"},
        {"id": "new2", "content": "fresh memory 2"},
    ]
    wm.bind_to_episodic_buffer(retrieved)
    eb = wm.get_component("episodic_buffer")
    assert len(eb) == 4
    contents = [item.content for item in eb]
    assert "old_0" not in contents
    assert "old_1" not in contents
    assert "fresh memory 1" in contents
    assert "fresh memory 2" in contents


def test_bind_to_episodic_buffer_empty_list():
    """bind_to_episodic_buffer with empty list is a no-op."""
    wm = WorkingMemory()
    wm.bind_to_episodic_buffer([])
    assert len(wm.get_component("episodic_buffer")) == 0


def test_bind_to_episodic_buffer_missing_fields():
    """bind_to_episodic_buffer handles memories with missing fields."""
    wm = WorkingMemory()
    retrieved = [{"content": "no id memory"}, {"id": "has_id"}]
    wm.bind_to_episodic_buffer(retrieved)
    eb = wm.get_component("episodic_buffer")
    assert len(eb) == 2
    assert eb[0].content == "no id memory"
    assert eb[1].content == ""  # missing content defaults to ""


# ── Cross-component operations ───────────────────────────────────────

def test_rehearse_finds_item_in_any_component():
    """Rehearse searches across all components."""
    wm = WorkingMemory()
    wm.load(WorkingMemoryItem(content="word", slot="phonological"))
    wm.load(WorkingMemoryItem(content="visual", slot="visuospatial"))
    wm.load(WorkingMemoryItem(content="bound", slot="episodic_buffer"))

    assert wm.rehearse("visual") is True
    assert wm.rehearse("bound") is True
    assert wm.rehearse("nonexistent") is False

    # Rehearsed items should have incremented reference counts
    vis = wm.get_component("visuospatial")
    assert vis[0].reference_count == 1
    eb = wm.get_component("episodic_buffer")
    assert eb[0].reference_count == 1


def test_session_boundary_across_components():
    """on_session_boundary filters items across all components."""
    wm = WorkingMemory()
    wm.load(WorkingMemoryItem(content="auth bug", slot="phonological"))
    wm.load(WorkingMemoryItem(content="weather chat", slot="phonological"))
    wm.load(WorkingMemoryItem(content="auth diagram", slot="visuospatial"))
    wm.load(WorkingMemoryItem(content="lunch note", slot="episodic_buffer"))

    evicted = wm.on_session_boundary(lambda item: "auth" in item.content)
    assert len(evicted) == 2  # weather + lunch
    remaining = wm.get_slots()
    assert len(remaining) == 2
    contents = [item.content for item in remaining]
    assert "auth bug" in contents
    assert "auth diagram" in contents


def test_clear_empties_all_components():
    """clear() removes items from every component."""
    wm = WorkingMemory()
    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="visuospatial"))
    wm.load(WorkingMemoryItem(content="c", slot="episodic_buffer"))
    assert len(wm.get_slots()) == 3
    wm.clear()
    assert len(wm.get_slots()) == 0
    assert len(wm.get_component("phonological")) == 0
    assert len(wm.get_component("visuospatial")) == 0
    assert len(wm.get_component("episodic_buffer")) == 0


def test_free_slots_sums_all_components():
    """free_slots returns the total free capacity across all components."""
    wm = WorkingMemory(capacity=4)
    # Total capacity: phonological(4) + visuospatial(3) + episodic_buffer(4) = 11
    assert wm.free_slots == 11

    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="visuospatial"))
    assert wm.free_slots == 9  # 3 + 2 + 4

    # Fill phonological completely
    for i in range(3):
        wm.load(WorkingMemoryItem(content=f"p{i}", slot="phonological"))
    assert wm.free_slots == 6  # 0 + 2 + 4


def test_get_context_includes_all_components():
    """get_context() joins content from all components."""
    wm = WorkingMemory()
    wm.load(WorkingMemoryItem(content="spoken word", slot="phonological"))
    wm.load(WorkingMemoryItem(content="spatial map", slot="visuospatial"))
    wm.load(WorkingMemoryItem(content="bound episode", slot="episodic_buffer"))
    ctx = wm.get_context()
    assert "spoken word" in ctx
    assert "spatial map" in ctx
    assert "bound episode" in ctx


def test_wm_item_with_metadata():
    wm = WorkingMemory(capacity=4)
    item = WorkingMemoryItem(
        content="find auth bug in login",
        slot="phonological",
        metadata={"intent": "command", "keywords": ["find", "auth", "bug", "login"]},
    )
    wm.load(item)
    slots = wm.get_slots()
    assert len(slots) == 1
    assert slots[0].metadata["intent"] == "command"
    assert "keywords" in slots[0].metadata


def test_get_slots_returns_all_items_across_components():
    """get_slots() returns items from all components in a single list."""
    wm = WorkingMemory()
    wm.load(WorkingMemoryItem(content="p1", slot="phonological"))
    wm.load(WorkingMemoryItem(content="v1", slot="visuospatial"))
    wm.load(WorkingMemoryItem(content="e1", slot="episodic_buffer"))
    all_items = wm.get_slots()
    assert len(all_items) == 3
    contents = {item.content for item in all_items}
    assert contents == {"p1", "v1", "e1"}
