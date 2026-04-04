from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NetworkMode(str, Enum):
    DMN = "default_mode"
    ECN = "executive_control"
    CREATIVE = "creative"


MODE_REGIONS: dict[NetworkMode, set[str]] = {
    NetworkMode.DMN: {"hippocampus", "consolidation"},
    NetworkMode.ECN: {
        "prefrontal_cortex", "acc", "basal_ganglia", "cerebellum",
        "thalamus", "wernicke", "broca",
        "visual_cortex", "auditory_cortex_l", "auditory_cortex_r",
        "angular_gyrus",
        # 7-Phase additions: dual-stream + integration + output
        "psts", "spt", "motor_cortex",
    },
    NetworkMode.CREATIVE: {"prefrontal_cortex", "hippocampus", "acc", "psts", "spt"},
}

ALWAYS_ACTIVE: set[str] = {
    "amygdala", "hypothalamus", "salience_network",
    "brainstem", "vta",
    "corpus_callosum",  # Inter-hemisphere communication always available
}


@dataclass
class NetworkSwitch:
    from_mode: NetworkMode
    to_mode: NetworkMode
    trigger: str = ""


class TripleNetworkController:
    def __init__(self):
        self.current_mode: NetworkMode = NetworkMode.DMN
        self.switch_history: list[NetworkSwitch] = []

    def switch_to(self, mode: NetworkMode, trigger: str = "") -> NetworkSwitch:
        switch = NetworkSwitch(from_mode=self.current_mode, to_mode=mode, trigger=trigger)
        self.current_mode = mode
        self.switch_history.append(switch)
        return switch

    def get_active_regions(self) -> set[str]:
        return MODE_REGIONS[self.current_mode] | ALWAYS_ACTIVE

    def is_region_active(self, region_name: str) -> bool:
        return region_name in self.get_active_regions()
