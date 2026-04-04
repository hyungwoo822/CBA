"""Hierarchical goal representation (Koechlin 2003).

Brain mapping: Rostro-caudal gradient in PFC.
  - Rostral (front): abstract, long-term goals
  - Mid: sub-goals, contextual rules
  - Caudal (back): immediate action plans

Data structure: Tree (parent-child), matching brain's hierarchical control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class GoalNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    level: str = "caudal"  # rostral | mid | caudal
    status: str = "active"  # active | completed | failed | suspended
    children: list[GoalNode] = field(default_factory=list)
    parent_id: Optional[str] = None

    def add_subgoal(self, description: str, level: str = "caudal") -> GoalNode:
        """Add a child goal under this node."""
        child = GoalNode(description=description, level=level, parent_id=self.id)
        self.children.append(child)
        return child

    def get_active_leaf(self) -> Optional[GoalNode]:
        """Get the deepest active goal (caudal = current action)."""
        for child in self.children:
            if child.status == "active":
                leaf = child.get_active_leaf()
                return leaf if leaf else child
        return self if self.status == "active" else None

    def complete(self) -> None:
        """Mark this goal and all active children as completed."""
        self.status = "completed"
        for child in self.children:
            if child.status == "active":
                child.complete()

    def to_context(self, depth: int = 0) -> str:
        """Serialize goal tree for LLM context."""
        indent = "  " * depth
        prefix = {"rostral": "[ABSTRACT]", "mid": "[SUB-GOAL]", "caudal": "[ACTION]"}
        status_mark = "x" if self.status == "completed" else " "
        line = f"{indent}[{status_mark}] {prefix.get(self.level, '')} {self.description}"
        lines = [line]
        for child in self.children:
            lines.append(child.to_context(depth + 1))
        return "\n".join(lines)


class GoalTree:
    """Manages the hierarchical goal stack for PFC."""

    def __init__(self):
        self.roots: list[GoalNode] = []

    def set_goal(self, description: str, level: str = "rostral") -> GoalNode:
        """Set a new top-level (rostral) goal."""
        node = GoalNode(description=description, level=level)
        self.roots.append(node)
        return node

    def get_current_focus(self) -> Optional[GoalNode]:
        """Get the most recent active leaf goal."""
        for root in reversed(self.roots):
            if root.status == "active":
                return root.get_active_leaf()
        return None

    def to_context(self) -> str:
        """Serialize all active goals for LLM prompt injection."""
        active = [r for r in self.roots if r.status == "active"]
        if not active:
            return "No active goals."
        return "\n".join(r.to_context() for r in active)

    def clear(self) -> None:
        """Clear all goals (used on strategy switch)."""
        self.roots.clear()
