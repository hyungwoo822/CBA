"""Tests for universal ontology seed constants."""
from brain_agent.memory.ontology_seed import (
    UNIVERSAL_NODE_TYPES,
    UNIVERSAL_RELATION_TYPES,
)


def test_seven_node_types():
    names = [node_type["name"] for node_type in UNIVERSAL_NODE_TYPES]
    assert len(names) == 7
    assert set(names) == {
        "Entity",
        "Person",
        "Artifact",
        "Event",
        "Concept",
        "Statement",
        "Source",
    }


def test_ten_relation_types():
    names = [relation_type["name"] for relation_type in UNIVERSAL_RELATION_TYPES]
    assert len(names) == 10
    assert set(names) == {
        "is_a",
        "part_of",
        "has_part",
        "refers_to",
        "happened_at",
        "said_by",
        "contradicts",
        "supersedes",
        "superseded_by",
        "derived_from",
    }


def test_node_parent_references_valid():
    names = {node_type["name"] for node_type in UNIVERSAL_NODE_TYPES}
    for node_type in UNIVERSAL_NODE_TYPES:
        parent = node_type.get("parent")
        if parent is not None:
            assert parent in names, f"{node_type['name']} parent {parent} not defined"


def test_person_extends_entity():
    person = next(nt for nt in UNIVERSAL_NODE_TYPES if nt["name"] == "Person")
    assert person["parent"] == "Entity"


def test_part_of_has_inverse():
    part_of = next(rt for rt in UNIVERSAL_RELATION_TYPES if rt["name"] == "part_of")
    has_part = next(rt for rt in UNIVERSAL_RELATION_TYPES if rt["name"] == "has_part")
    assert part_of["inverse_of"] == "has_part"
    assert has_part["inverse_of"] == "part_of"


def test_supersedes_has_inverse():
    supersedes = next(
        rt for rt in UNIVERSAL_RELATION_TYPES if rt["name"] == "supersedes"
    )
    superseded_by = next(
        rt for rt in UNIVERSAL_RELATION_TYPES if rt["name"] == "superseded_by"
    )
    assert supersedes["inverse_of"] == "superseded_by"
    assert superseded_by["inverse_of"] == "supersedes"


def test_contradicts_is_symmetric():
    contradicts = next(
        rt for rt in UNIVERSAL_RELATION_TYPES if rt["name"] == "contradicts"
    )
    assert contradicts["symmetric"] is True


def test_event_requires_happened_at():
    event = next(nt for nt in UNIVERSAL_NODE_TYPES if nt["name"] == "Event")
    assert "happened_at" in event["schema"]["required"]
