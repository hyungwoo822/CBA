"""Pure-data tests for bundled ontology templates."""
import pytest

from brain_agent.memory.templates import (
    get_template,
    list_templates,
    personal_knowledge,
    research_notes,
    software_project,
)


def test_software_project_meta():
    template = software_project.TEMPLATE
    assert template["name"] == "software-project"
    assert template["version"] == "1.0"
    assert template["decay_policy"] == "none"
    assert software_project.VERSION == "1.0"


def test_software_project_node_types():
    node_names = [nt["name"] for nt in software_project.TEMPLATE["node_types"]]
    assert len(node_names) == 10
    assert set(node_names) == {
        "Requirement",
        "Decision",
        "Module",
        "Interface",
        "Constraint",
        "Risk",
        "NonGoal",
        "Workflow",
        "DomainTerm",
        "DataModel",
    }


def test_software_project_parents_reference_universal_or_local():
    allowed_parents = {
        "Entity",
        "Person",
        "Artifact",
        "Event",
        "Concept",
        "Statement",
        "Source",
        "Requirement",
        "Decision",
        "Module",
        "Interface",
        "Constraint",
        "Risk",
        "NonGoal",
        "Workflow",
        "DomainTerm",
        "DataModel",
    }
    for node_type in software_project.TEMPLATE["node_types"]:
        parent = node_type.get("parent")
        if parent is not None:
            assert parent in allowed_parents


def test_software_project_relation_types():
    names = [rt["name"] for rt in software_project.TEMPLATE["relation_types"]]
    assert len(names) == 10
    assert set(names) == {
        "depends_on",
        "implements",
        "constrains",
        "blocks",
        "trades_off_against",
        "belongs_to",
        "conflicts_with",
        "mitigates",
        "exposes",
        "stores",
    }


def test_software_project_implements_domain_range():
    relation = next(
        rt
        for rt in software_project.TEMPLATE["relation_types"]
        if rt["name"] == "implements"
    )
    assert relation["domain"] == "Module"
    assert relation["range"] == "Requirement"


def test_software_project_trades_off_symmetric():
    relation = next(
        rt
        for rt in software_project.TEMPLATE["relation_types"]
        if rt["name"] == "trades_off_against"
    )
    assert relation["symmetric"] is True


def test_software_project_conflicts_with_symmetric():
    relation = next(
        rt
        for rt in software_project.TEMPLATE["relation_types"]
        if rt["name"] == "conflicts_with"
    )
    assert relation["symmetric"] is True


def test_software_project_depends_on_transitive():
    relation = next(
        rt
        for rt in software_project.TEMPLATE["relation_types"]
        if rt["name"] == "depends_on"
    )
    assert relation["transitive"] is True


def test_research_notes_meta():
    template = research_notes.TEMPLATE
    assert template["name"] == "research-notes"
    assert template["version"] == "1.0"
    assert template["decay_policy"] == "none"
    assert research_notes.VERSION == "1.0"


def test_research_notes_node_types():
    names = [nt["name"] for nt in research_notes.TEMPLATE["node_types"]]
    assert len(names) == 6
    assert set(names) == {
        "Hypothesis",
        "Experiment",
        "Result",
        "Citation",
        "Finding",
        "Methodology",
    }


def test_research_notes_citation_parent_is_source():
    node_type = next(
        nt for nt in research_notes.TEMPLATE["node_types"] if nt["name"] == "Citation"
    )
    assert node_type["parent"] == "Source"


def test_research_notes_relation_types():
    names = [rt["name"] for rt in research_notes.TEMPLATE["relation_types"]]
    assert len(names) == 6
    assert set(names) == {
        "tests",
        "supports",
        "refutes",
        "cites",
        "replicates",
        "extends",
    }


def test_research_notes_replicates_domain_range():
    relation = next(
        rt
        for rt in research_notes.TEMPLATE["relation_types"]
        if rt["name"] == "replicates"
    )
    assert relation["domain"] == "Experiment"
    assert relation["range"] == "Experiment"


def test_research_notes_tests_wires_experiment_to_hypothesis():
    relation = next(
        rt for rt in research_notes.TEMPLATE["relation_types"] if rt["name"] == "tests"
    )
    assert relation["domain"] == "Experiment"
    assert relation["range"] == "Hypothesis"


def test_personal_knowledge_meta():
    template = personal_knowledge.TEMPLATE
    assert template["name"] == "personal-knowledge"
    assert template["version"] == "1.0"
    assert template["decay_policy"] == "normal"
    assert personal_knowledge.VERSION == "1.0"


def test_personal_knowledge_node_types():
    names = [nt["name"] for nt in personal_knowledge.TEMPLATE["node_types"]]
    assert len(names) == 5
    assert set(names) == {"Preference", "Habit", "Belief", "Memory", "Goal"}


def test_personal_knowledge_relation_types():
    names = [rt["name"] for rt in personal_knowledge.TEMPLATE["relation_types"]]
    assert len(names) == 5
    assert set(names) == {
        "prefers_over",
        "causes",
        "reminds_of",
        "wants",
        "knows",
    }


def test_personal_knowledge_wants_person_to_goal():
    relation = next(
        rt
        for rt in personal_knowledge.TEMPLATE["relation_types"]
        if rt["name"] == "wants"
    )
    assert relation["domain"] == "Person"
    assert relation["range"] == "Goal"


def test_personal_knowledge_reminds_of_symmetric():
    relation = next(
        rt
        for rt in personal_knowledge.TEMPLATE["relation_types"]
        if rt["name"] == "reminds_of"
    )
    assert relation["symmetric"] is True


def test_personal_knowledge_causes_transitive():
    relation = next(
        rt
        for rt in personal_knowledge.TEMPLATE["relation_types"]
        if rt["name"] == "causes"
    )
    assert relation["transitive"] is True


def test_get_template_known():
    assert get_template("software-project")["name"] == "software-project"
    assert get_template("research-notes")["name"] == "research-notes"
    assert get_template("personal-knowledge")["name"] == "personal-knowledge"


def test_get_template_unknown_raises():
    with pytest.raises(ValueError, match="Unknown template"):
        get_template("no-such-template")


def test_list_templates_returns_all_three():
    items = list_templates()
    assert len(items) == 3
    names = {item["name"] for item in items}
    assert names == {"software-project", "research-notes", "personal-knowledge"}
    for item in items:
        assert item["version"] == "1.0"
        assert item["node_type_count"] > 0
        assert item["relation_type_count"] > 0
