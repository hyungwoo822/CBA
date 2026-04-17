"""Phase 1 end-to-end smoke tests through MemoryManager."""
import aiosqlite


async def test_memory_manager_exposes_raw_vault(memory_manager):
    from brain_agent.memory.raw_vault import RawVault

    assert isinstance(memory_manager.raw_vault, RawVault)


async def test_migration_m001_applied_automatically(tmp_path, mock_embedding):
    from brain_agent.memory.manager import MemoryManager

    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        async with aiosqlite.connect(str(tmp_path / "brain_state.db")) as db:
            async with db.execute(
                "SELECT migration_id FROM schema_version "
                "WHERE migration_id = 'm001_workspace_columns'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
    finally:
        await mm.close()


async def test_raw_vault_ingest_roundtrip(memory_manager):
    src = await memory_manager.raw_vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        extracted_text="hello from smoke",
    )
    got = await memory_manager.raw_vault.get_raw_bytes(src["id"])
    assert got == b"hello from smoke"


async def test_backward_compat_existing_semantic_calls_land_personal(
    memory_manager,
):
    await memory_manager.semantic.add_relationship("x", "rel", "y")
    rels = await memory_manager.semantic.get_relationships("x")
    assert rels[0]["workspace_id"] == "personal"


async def test_cross_workspace_edge_end_to_end(memory_manager):
    ws = await memory_manager.workspace.create_workspace(name="CrossProj")
    await memory_manager.semantic.add_relationship(
        "alice",
        "references",
        "crossproj_spec",
        workspace_id="personal",
        target_workspace_id=ws["id"],
    )
    graph = await memory_manager.semantic.export_as_networkx(
        workspace_id="personal", include_cross_refs=True
    )
    assert graph.has_edge("alice", "crossproj_spec")
    graph2 = await memory_manager.semantic.export_as_networkx(
        workspace_id=ws["id"], include_cross_refs=True
    )
    assert graph2.has_edge("alice", "crossproj_spec")


async def test_reopen_preserves_vault_and_migrations(tmp_path, mock_embedding):
    from brain_agent.memory.manager import MemoryManager

    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        src = await mm.raw_vault.ingest(
            workspace_id="personal",
            kind="user_utterance",
            extracted_text="persistent",
        )
        src_id = src["id"]
    finally:
        await mm.close()

    mm2 = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm2.initialize()
    try:
        back = await mm2.raw_vault.get_raw_bytes(src_id)
        assert back == b"persistent"
    finally:
        await mm2.close()
