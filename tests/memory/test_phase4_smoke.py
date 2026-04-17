"""Phase 4 smoke tests for the existing identity_facts path.

PersonalAdapter is additive. Existing direct SemanticStore identity_facts
callers must keep working after MemoryManager exposes the adapter.
"""


async def test_legacy_semantic_identity_facts_api_still_works(memory_manager):
    """Existing direct semantic_store callers are unaffected by PersonalAdapter."""
    await memory_manager.semantic.add_identity_fact(
        "user_model", "name", "Alice", source="legacy", confidence=0.9,
    )
    await memory_manager.semantic.add_identity_fact(
        "self_model", "role", "assistant", source="legacy", confidence=1.0,
    )

    user_facts = await memory_manager.semantic.get_identity_facts("user_model")
    self_facts = await memory_manager.semantic.get_identity_facts("self_model")

    assert len(user_facts) == 1
    assert user_facts[0]["key"] == "name"
    assert user_facts[0]["value"] == "Alice"
    assert user_facts[0]["source"] == "legacy"
    assert set(user_facts[0].keys()) == {
        "key",
        "value",
        "source",
        "confidence",
        "updated_at",
    }

    assert len(self_facts) == 1
    assert self_facts[0]["value"] == "assistant"


async def test_render_user_context_still_reads_identity_facts(memory_manager):
    """MemoryManager.render_user_context still reads direct identity_facts."""
    await memory_manager.semantic.add_identity_fact(
        "user_model",
        "preference:coffee",
        "loves espresso",
        source="legacy",
    )
    markdown = await memory_manager.render_user_context()
    assert markdown != ""
    assert "User Profile" in markdown
    assert "espresso" in markdown


async def test_adapter_round_trip_visible_via_legacy_api(memory_manager):
    """Writes through PersonalAdapter are visible to get_identity_facts."""
    await memory_manager.personal.add_user_fact("city", "Seoul", confidence=0.8)

    legacy_view = await memory_manager.semantic.get_identity_facts("user_model")
    assert len(legacy_view) == 1
    assert legacy_view[0]["key"] == "city"
    assert legacy_view[0]["value"] == "Seoul"


async def test_adapter_write_from_nodes_visible_via_legacy_api(memory_manager):
    """write_from_nodes output is observable to pre-Phase-4 callers."""
    nodes = [
        {
            "type": "Person",
            "label": "user",
            "workspace_id": "personal",
            "properties": {"name": "Alice"},
        },
    ]
    await memory_manager.personal.write_from_nodes(nodes)
    legacy_view = await memory_manager.semantic.get_identity_facts("user_model")
    assert legacy_view[0]["value"] == "Alice"
