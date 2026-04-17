"""Research-notes domain ontology template."""
from __future__ import annotations


VERSION = "1.0"

TEMPLATE: dict = {
    "name": "research-notes",
    "version": VERSION,
    "description": "Research and academic knowledge",
    "decay_policy": "none",
    "node_types": [
        {
            "name": "Hypothesis",
            "parent": "Statement",
            "schema": {
                "props": ["testable", "status"],
                "required": [],
            },
        },
        {
            "name": "Experiment",
            "parent": "Event",
            "schema": {
                "props": ["method", "sample_size"],
                "required": [],
            },
        },
        {
            "name": "Result",
            "parent": "Statement",
            "schema": {
                "props": ["p_value", "effect_size"],
                "required": [],
            },
        },
        {
            "name": "Citation",
            "parent": "Source",
            "schema": {
                "props": ["authors", "year", "doi"],
                "required": [],
            },
        },
        {
            "name": "Finding",
            "parent": "Statement",
            "schema": {
                "props": ["replicable"],
                "required": [],
            },
        },
        {
            "name": "Methodology",
            "parent": "Concept",
            "schema": {
                "props": ["strengths", "limitations"],
                "required": [],
            },
        },
    ],
    "relation_types": [
        {
            "name": "tests",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Experiment",
            "range": "Hypothesis",
        },
        {
            "name": "supports",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Result",
            "range": "Hypothesis",
        },
        {
            "name": "refutes",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Result",
            "range": "Hypothesis",
        },
        {
            "name": "cites",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "replicates",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Experiment",
            "range": "Experiment",
        },
        {
            "name": "extends",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
    ],
}
