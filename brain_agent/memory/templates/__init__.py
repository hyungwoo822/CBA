"""Pre-built domain ontology templates."""
from __future__ import annotations

import importlib


_MODULE_BY_NAME: dict[str, str] = {
    "software-project": "software_project",
    "research-notes": "research_notes",
    "personal-knowledge": "personal_knowledge",
}


def get_template(name: str) -> dict:
    """Return the TEMPLATE dict for a bundled template name."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise ValueError(f"Unknown template: {name}")
    module = importlib.import_module(f"brain_agent.memory.templates.{module_name}")
    return module.TEMPLATE


def list_templates() -> list[dict]:
    """Return metadata for every bundled template."""
    result: list[dict] = []
    for module_name in _MODULE_BY_NAME.values():
        module = importlib.import_module(f"brain_agent.memory.templates.{module_name}")
        template = module.TEMPLATE
        result.append(
            {
                "name": template["name"],
                "version": template["version"],
                "description": template.get("description", ""),
                "decay_policy": template.get("decay_policy", "normal"),
                "node_type_count": len(template.get("node_types", [])),
                "relation_type_count": len(template.get("relation_types", [])),
            }
        )
    return result


__all__ = ["get_template", "list_templates"]
