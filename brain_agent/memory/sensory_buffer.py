from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass
class SensoryItem:
    data: dict
    modality: str

class SensoryBuffer:
    def __init__(self):
        self._items: list[SensoryItem] = []
    def register(self, data: dict, modality: str = "text") -> None:
        self._items.append(SensoryItem(data=data, modality=modality))
    def get_all(self) -> list[SensoryItem]:
        return list(self._items)
    def attend(self, filter_fn: Callable[[SensoryItem], bool]) -> list[SensoryItem]:
        return [item for item in self._items if filter_fn(item)]
    def flush(self) -> None:
        self._items.clear()
    def new_cycle(self) -> None:
        self.flush()
