"""Fixture: software-project v2.0 major bump."""
from __future__ import annotations


VERSION = "2.0"

TEMPLATE: dict = {
    "name": "software-project",
    "version": VERSION,
    "description": "Software engineering project knowledge v2",
    "decay_policy": "none",
    "node_types": [
        {
            "name": "Requirement",
            "parent": "Statement",
            "schema": {
                "props": ["priority", "status", "acceptance_criteria"],
                "required": ["priority"],
            },
        },
        {
            "name": "Decision",
            "parent": "Statement",
            "schema": {
                "props": ["date", "rationale", "alternatives", "decided_by"],
                "required": ["rationale"],
            },
        },
        {
            "name": "Module",
            "parent": "Artifact",
            "schema": {"props": ["path", "language", "version"], "required": []},
        },
        {
            "name": "Interface",
            "parent": "Artifact",
            "schema": {
                "props": ["protocol", "spec_url", "auth_method"],
                "required": [],
            },
        },
        {
            "name": "Constraint",
            "parent": "Statement",
            "schema": {
                "props": ["hard_or_soft", "metric", "threshold"],
                "required": [],
            },
        },
        {
            "name": "Risk",
            "parent": "Concept",
            "schema": {
                "props": ["likelihood", "impact", "mitigation"],
                "required": [],
            },
        },
        {
            "name": "Workflow",
            "parent": "Event",
            "schema": {"props": ["trigger", "steps", "output"], "required": []},
        },
        {
            "name": "DomainTerm",
            "parent": "Concept",
            "schema": {
                "props": ["definition", "examples"],
                "required": ["definition"],
            },
        },
        {
            "name": "DataModel",
            "parent": "Artifact",
            "schema": {
                "props": ["fields", "constraints", "storage"],
                "required": [],
            },
        },
    ],
    "relation_types": [
        {
            "name": "depends_on",
            "transitive": True,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "implements",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Module",
            "range": "Requirement",
        },
        {
            "name": "constrains",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Constraint",
            "range": None,
        },
        {
            "name": "blocks",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "trades_off_against",
            "transitive": False,
            "symmetric": True,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "belongs_to",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": None,
            "range": "Module",
        },
        {
            "name": "conflicts_with",
            "transitive": False,
            "symmetric": True,
            "inverse_of": None,
            "domain": None,
            "range": None,
        },
        {
            "name": "mitigates",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Artifact",
            "range": "Risk",
        },
        {
            "name": "exposes",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Module",
            "range": "Interface",
        },
        {
            "name": "stores",
            "transitive": False,
            "symmetric": False,
            "inverse_of": None,
            "domain": "Module",
            "range": "DataModel",
        },
    ],
}
