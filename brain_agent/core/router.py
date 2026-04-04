from __future__ import annotations
from dataclasses import dataclass
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController
from brain_agent.core.neuromodulators import Neuromodulators

FORCE_ECN_TYPES = {SignalType.CONFLICT_DETECTED, SignalType.STRATEGY_SWITCH}

ECN_ROUTES = {
    SignalType.EXTERNAL_INPUT: ["thalamus", "salience_network"],
    SignalType.TEXT_INPUT: ["thalamus", "wernicke", "salience_network"],
    SignalType.PLAN: ["acc", "basal_ganglia"],
    SignalType.ACTION_SELECTED: ["cerebellum"],
    SignalType.ACTION_RESULT: ["cerebellum", "acc", "hippocampus", "amygdala"],
    SignalType.CONFLICT_DETECTED: ["prefrontal_cortex"],
    SignalType.STRATEGY_SWITCH: ["prefrontal_cortex"],
    SignalType.PREDICTION_ERROR: ["acc"],
    SignalType.EMOTIONAL_TAG: ["hippocampus"],
    SignalType.CONSOLIDATION_TRIGGER: ["consolidation"],
    SignalType.RESOURCE_STATUS: ["hypothalamus"],
    SignalType.ENCODE: ["hippocampus"],
    SignalType.RETRIEVE: ["hippocampus"],
    SignalType.IMAGE_INPUT: ["visual_cortex", "angular_gyrus", "thalamus", "salience_network"],
    SignalType.AUDIO_INPUT: ["auditory_cortex_l", "auditory_cortex_r", "spt", "thalamus", "salience_network"],
}
DMN_ROUTES = {
    SignalType.EXTERNAL_INPUT: ["thalamus", "salience_network"],
    SignalType.TEXT_INPUT: ["thalamus", "wernicke", "salience_network"],
    SignalType.ACTION_RESULT: ["hippocampus", "amygdala"],
    SignalType.CONFLICT_DETECTED: ["prefrontal_cortex"],
    SignalType.STRATEGY_SWITCH: ["prefrontal_cortex"],
    SignalType.EMOTIONAL_TAG: ["hippocampus"],
    SignalType.CONSOLIDATION_TRIGGER: ["consolidation"],
    SignalType.RESOURCE_STATUS: ["hypothalamus"],
    SignalType.ENCODE: ["hippocampus"],
    SignalType.RETRIEVE: ["hippocampus"],
    SignalType.IMAGE_INPUT: ["visual_cortex", "angular_gyrus", "thalamus", "salience_network"],
    SignalType.AUDIO_INPUT: ["auditory_cortex_l", "auditory_cortex_r", "spt", "thalamus", "salience_network"],
}


@dataclass
class RoutingEvent:
    source: str
    targets: list[str]
    signal_type: str
    priority: float


class ThalamicRouter:
    def __init__(self, network_ctrl: TripleNetworkController, neuromodulators: Neuromodulators):
        self._network_ctrl = network_ctrl
        self._neuromod = neuromodulators
        self.event_log: list[RoutingEvent] = []

    def resolve_targets(self, signal: Signal) -> list[str]:
        mode = self._network_ctrl.current_mode
        if signal.type == SignalType.GWT_BROADCAST:
            targets = list(self._network_ctrl.get_active_regions())
        elif signal.type in FORCE_ECN_TYPES:
            targets = ECN_ROUTES.get(signal.type, [])
        elif mode in (NetworkMode.ECN, NetworkMode.CREATIVE):
            targets = ECN_ROUTES.get(signal.type, [])
        else:
            targets = DMN_ROUTES.get(signal.type, [])
        self.event_log.append(RoutingEvent(
            source=signal.source, targets=targets,
            signal_type=signal.type.value, priority=self.compute_priority(signal),
        ))
        return targets

    def compute_priority(self, signal: Signal) -> float:
        base = signal.priority
        arousal_boost = signal.emotional_tag.arousal * 0.5 if signal.emotional_tag else 0.0
        return base * (1.0 + arousal_boost) * (0.5 + self._neuromod.urgency * 0.5)
