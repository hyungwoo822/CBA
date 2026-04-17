"""Phase 0 end-to-end smoke tests through MemoryManager."""
from brain_agent.memory.ontology_seed import UNIVERSAL_WORKSPACE_ID
from brain_agent.memory.workspace_store import PERSONAL_WORKSPACE_ID


async def test_end_to_end_workspace_and_ontology(memory_manager):
    ws = await memory_manager.workspace.create_workspace(
        name="Billing Service",
        description="Payment flow",
        decay_policy="none",
    )
    endpoint_type = await memory_manager.ontology.register_node_type(
        ws["id"],
        "ApiEndpoint",
        parent_name="Artifact",
        source_snippet="/payments endpoint",
    )
    assert endpoint_type["confidence"] == "PROVISIONAL"

    await memory_manager.ontology.increment_occurrence(endpoint_type["id"])
    result = await memory_manager.ontology.increment_occurrence(endpoint_type["id"])
    assert result["confidence"] == "STABLE"
    assert result["promoted"] is True

    await memory_manager.workspace.set_session_workspace("smoke-session", ws["id"])
    assert (
        await memory_manager.workspace.get_session_workspace("smoke-session")
        == ws["id"]
    )

    all_types = await memory_manager.ontology.get_node_types(ws["id"])
    names = {node_type["name"] for node_type in all_types}
    assert "Artifact" in names
    assert "ApiEndpoint" in names

    personal = await memory_manager.workspace.get_workspace(PERSONAL_WORKSPACE_ID)
    assert personal["name"] == "Personal Knowledge"


async def test_universal_seed_survives_reopen(tmp_path, mock_embedding):
    from brain_agent.memory.manager import MemoryManager

    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    first_count = len(await mm.ontology.get_node_types(UNIVERSAL_WORKSPACE_ID))
    await mm.close()

    mm2 = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm2.initialize()
    second_count = len(await mm2.ontology.get_node_types(UNIVERSAL_WORKSPACE_ID))
    await mm2.close()

    assert first_count == second_count == 7
