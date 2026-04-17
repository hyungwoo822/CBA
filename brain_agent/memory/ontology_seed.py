"""Universal ontology seed visible to every workspace."""
from __future__ import annotations


UNIVERSAL_WORKSPACE_ID = "__universal__"

UNIVERSAL_NODE_TYPES: list[dict] = [
    {
        "name": "Entity",
        "parent": None,
        "schema": {"props": [], "required": []},
        "description": "Root type for tangible things",
    },
    {
        "name": "Person",
        "parent": "Entity",
        "schema": {"props": [], "required": []},
        "description": "A human actor or stakeholder",
    },
    {
        "name": "Artifact",
        "parent": "Entity",
        "schema": {"props": [], "required": []},
        "description": "A created object such as code, document, or tool",
    },
    {
        "name": "Event",
        "parent": None,
        "schema": {"props": ["happened_at", "actor"], "required": ["happened_at"]},
        "description": "A temporal occurrence",
    },
    {
        "name": "Concept",
        "parent": None,
        "schema": {"props": [], "required": []},
        "description": "An abstract idea or domain term",
    },
    {
        "name": "Statement",
        "parent": None,
        "schema": {"props": ["asserter", "confidence"], "required": []},
        "description": "A claim, decision, or assertion",
    },
    {
        "name": "Source",
        "parent": None,
        "schema": {"props": ["uri", "sha256", "kind"], "required": []},
        "description": "Provenance pointer to original input",
    },
]

UNIVERSAL_RELATION_TYPES: list[dict] = [
    {
        "name": "is_a",
        "transitive": True,
        "symmetric": False,
        "inverse_of": None,
        "domain": None,
        "range": None,
        "description": "Type-subtype hierarchy",
    },
    {
        "name": "part_of",
        "transitive": True,
        "symmetric": False,
        "inverse_of": "has_part",
        "domain": None,
        "range": None,
        "description": "Mereological containment",
    },
    {
        "name": "has_part",
        "transitive": True,
        "symmetric": False,
        "inverse_of": "part_of",
        "domain": None,
        "range": None,
        "description": "Inverse of part_of",
    },
    {
        "name": "refers_to",
        "transitive": False,
        "symmetric": False,
        "inverse_of": None,
        "domain": None,
        "range": None,
        "description": "Generic reference or pointer",
    },
    {
        "name": "happened_at",
        "transitive": False,
        "symmetric": False,
        "inverse_of": None,
        "domain": "Event",
        "range": None,
        "description": "Temporal anchoring",
    },
    {
        "name": "said_by",
        "transitive": False,
        "symmetric": False,
        "inverse_of": None,
        "domain": "Statement",
        "range": None,
        "description": "Attribution of a claim",
    },
    {
        "name": "contradicts",
        "transitive": False,
        "symmetric": True,
        "inverse_of": None,
        "domain": None,
        "range": None,
        "description": "Two statements that cannot both be true",
    },
    {
        "name": "supersedes",
        "transitive": True,
        "symmetric": False,
        "inverse_of": "superseded_by",
        "domain": None,
        "range": None,
        "description": "Newer version replaces older",
    },
    {
        "name": "superseded_by",
        "transitive": True,
        "symmetric": False,
        "inverse_of": "supersedes",
        "domain": None,
        "range": None,
        "description": "Inverse of supersedes",
    },
    {
        "name": "derived_from",
        "transitive": True,
        "symmetric": False,
        "inverse_of": None,
        "domain": None,
        "range": None,
        "description": "Causal or logical derivation",
    },
]
