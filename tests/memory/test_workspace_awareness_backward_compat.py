"""Workspace-aware signature compatibility tests."""

import inspect


def test_graph_analysis_export_accepts_workspace_id():
    from brain_agent.memory import semantic_store

    sig = inspect.signature(semantic_store.SemanticStore.export_as_networkx)
    assert "workspace_id" in sig.parameters
    assert sig.parameters["workspace_id"].default is None


def test_graph_analysis_export_accepts_include_cross_refs():
    from brain_agent.memory import semantic_store

    sig = inspect.signature(semantic_store.SemanticStore.export_as_networkx)
    assert "include_cross_refs" in sig.parameters
    assert sig.parameters["include_cross_refs"].default is True


def test_staging_encode_accepts_workspace_id():
    from brain_agent.memory import hippocampal_staging

    sig = inspect.signature(hippocampal_staging.HippocampalStaging.encode)
    assert "workspace_id" in sig.parameters
    assert sig.parameters["workspace_id"].default == "personal"


def test_staging_get_unconsolidated_accepts_workspace_id():
    from brain_agent.memory import hippocampal_staging

    sig = inspect.signature(hippocampal_staging.HippocampalStaging.get_unconsolidated)
    assert "workspace_id" in sig.parameters
    assert sig.parameters["workspace_id"].default == "personal"
