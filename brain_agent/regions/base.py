from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from brain_agent.core.signals import Signal

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider


@dataclass
class Vec3:
    x: float
    y: float
    z: float


class Lobe(str, Enum):
    FRONTAL = "frontal"
    TEMPORAL = "temporal"
    PARIETAL = "parietal"
    OCCIPITAL = "occipital"
    INSULAR = "insular"
    DIENCEPHALON = "diencephalon"
    MIDBRAIN = "midbrain"
    BRAINSTEM = "brainstem"
    CEREBELLUM = "cerebellum"
    SUBCORTICAL = "subcortical"


class Hemisphere(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"


class BrainRegion(ABC):
    def __init__(
        self,
        name: str,
        position: Vec3 | None = None,
        lobe: Lobe = Lobe.SUBCORTICAL,
        hemisphere: Hemisphere = Hemisphere.BILATERAL,
        llm_provider: LLMProvider | None = None,
    ):
        self.name = name
        self.position = position or Vec3(0, 0, 0)
        self.lobe = lobe
        self.hemisphere = hemisphere
        self.llm_provider = llm_provider
        self.activation_level: float = 0.0
        self._events: list[dict] = []

    @abstractmethod
    async def process(self, signal: Signal) -> Signal | None: ...

    def emit_activation(self, level: float):
        self.activation_level = max(0.0, min(1.0, level))
        self._events.append({"region": self.name, "level": self.activation_level})
