"""Multi-component working memory based on Baddeley (2000).

Brain mapping:
  - Phonological loop: text/language processing (left temporal + Broca's area)
  - Visuospatial sketchpad: spatial/visual context (right parietal)
  - Episodic buffer: cross-modal integration + LTM fragments (frontal)
  - Central executive: attention allocation (dorsolateral PFC) — orchestrates
    the above, not a storage component itself.

Data structure: Dict of bounded lists (one per component), matching
separate neural substrates for each buffer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

# Per-component capacity limits (Baddeley 2000)
SLOT_CAPACITY: dict[str, int] = {
    "phonological": 4,      # Text/language
    "visuospatial": 3,      # Spatial/visual context
    "episodic_buffer": 4,   # Cross-modal integration + LTM fragments
}


@dataclass
class WorkingMemoryItem:
    content: str
    slot: str
    reference_count: int = 0
    linked_memories: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class WorkingMemory:
    """Baddeley multi-component working memory with 3 storage buffers.

    The ``capacity`` constructor parameter overrides the default phonological
    capacity, keeping backward compatibility with code that creates
    ``WorkingMemory(capacity=N)`` and loads items with slot="phonological".
    """

    def __init__(self, capacity: int = 4):
        self._default_capacity = capacity
        # Override phonological capacity with the constructor parameter
        self._capacities: dict[str, int] = dict(SLOT_CAPACITY)
        self._capacities["phonological"] = capacity

        # Three storage components
        self._slots: dict[str, list[WorkingMemoryItem]] = {
            "phonological": [],
            "visuospatial": [],
            "episodic_buffer": [],
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def load(self, item: WorkingMemoryItem) -> list[WorkingMemoryItem]:
        """Load an item into the appropriate component buffer.

        Items whose ``slot`` does not match a known component are routed
        to the phonological loop (backward-compatible default).

        Returns any items that were evicted to enforce the capacity limit.
        """
        slot_name = item.slot if item.slot in self._slots else "phonological"
        component = self._slots[slot_name]
        cap = self._capacities.get(slot_name, self._default_capacity)
        evicted: list[WorkingMemoryItem] = []
        component.append(item)
        while len(component) > cap:
            evicted.append(component.pop(0))
        return evicted

    def rehearse(self, content: str) -> bool:
        """Rehearse an item (search across all components).

        Moves the matching item to the end of its component list and
        increments its reference count.
        """
        for component in self._slots.values():
            for i, item in enumerate(component):
                if item.content == content:
                    item.reference_count += 1
                    component.append(component.pop(i))
                    return True
        return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_slots(self) -> list[WorkingMemoryItem]:
        """Return all items across all components (backward-compatible)."""
        items: list[WorkingMemoryItem] = []
        for component in self._slots.values():
            items.extend(component)
        return items

    def get_component(self, name: str) -> list[WorkingMemoryItem]:
        """Return items from a specific component."""
        return list(self._slots.get(name, []))

    def get_context(self) -> str:
        """Build context string from all components, including semantic metadata.

        When a WorkingMemoryItem carries metadata (e.g. Wernicke comprehension),
        the context string includes intent and keywords so downstream regions
        (PFC, retrieval) see the processed representation, not just raw text.
        """
        parts: list[str] = []
        for item in self.get_slots():
            meta = item.metadata
            if meta and meta.get("intent"):
                line = f"[{meta['intent']}] {item.content}"
                kw = meta.get("keywords")
                if kw:
                    line += f" (keywords: {', '.join(kw[:5])})"
                parts.append(line)
            else:
                parts.append(item.content)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_session_boundary(
        self, relevance_fn: Callable[[WorkingMemoryItem], bool]
    ) -> list[WorkingMemoryItem]:
        """Apply relevance filtering across all components."""
        evicted: list[WorkingMemoryItem] = []
        for name in self._slots:
            component = self._slots[name]
            keep = [item for item in component if relevance_fn(item)]
            removed = [item for item in component if not relevance_fn(item)]
            evicted.extend(removed)
            self._slots[name] = keep
        return evicted

    def clear(self) -> None:
        """Clear all components."""
        for component in self._slots.values():
            component.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def free_slots(self) -> int:
        """Sum of free slots across all components."""
        total_free = 0
        for name, component in self._slots.items():
            cap = self._capacities.get(name, self._default_capacity)
            total_free += max(0, cap - len(component))
        return total_free

    # ------------------------------------------------------------------
    # Episodic buffer integration (Baddeley 2000)
    # ------------------------------------------------------------------

    def bind_to_episodic_buffer(self, retrieved_memories: list[dict]) -> None:
        """Load retrieved LTM fragments into the episodic buffer.

        The episodic buffer serves as a cross-modal integration space that
        binds information from LTM with current working memory contents.
        Capacity-limited: only the first N fragments fit.
        """
        cap = self._capacities.get("episodic_buffer", 4)
        for mem in retrieved_memories[:cap]:
            item = WorkingMemoryItem(
                content=mem.get("content", ""),
                slot="episodic_buffer",
                linked_memories=[mem.get("id", "")],
            )
            self.load(item)
