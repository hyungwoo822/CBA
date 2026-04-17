"""Personal-knowledge domain ontology template."""
from __future__ import annotations


VERSION = "1.0"

TEMPLATE: dict = {
    "name": "personal-knowledge",
    "version": VERSION,
    "description": "Personal life knowledge and preferences",
    "decay_policy": "normal",
    "node_types": [
        {
            "name": "Preference",
            "parent": "Concept",
            "schema": {
                "props": ["strength", "since"],
                "required": [],
            },
        },
        {
            "name": "Habit",
            "parent": "Event",
            "schema": {
                "props": ["frequency", "trigger"],
                "required": [],
            },
        },
        {
            "name": "Belief",
            "parent": "Statement",
            "schema": {
                "props": ["certainty"],
                "required": [],
            },
        },
        {
            "name": "Memory",
            "parent": "Event",
            "schema": {
                "props": ["when", "where", "who"],
                "required": [],
            },
        },
        {
            "name": "Goal",
            "parent": "Statement",
            "schema": {
                "props": ["deadline", "progress"],
                "required": [],
            },
        },
    ],
    "relation_types": [
        {
            "name": "prefers_over",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "causes",
            "transitive": True,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "reminds_of",
            "transitive": False,
            "symmetric": True,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "wants",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Person",
            "range": "Goal",
        },
        {
            "name": "knows",
            "transitive": False,
            "symmetric": True,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
    ],
}
