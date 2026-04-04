from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController

ACTIVATION_THRESHOLD = 0.4
DEACTIVATION_THRESHOLD = 0.1


class SalienceNetworkRegion(BrainRegion):
    def __init__(self, network_ctrl: TripleNetworkController):
        super().__init__(name="salience_network", position=Vec3(30, 20, 10), lobe=Lobe.INSULAR, hemisphere=Hemisphere.BILATERAL)
        self._network_ctrl = network_ctrl
        self._last_novelty: float = 0.5

    async def process(self, signal: Signal) -> Signal | None:
        salience = self._compute_salience(signal)
        self.emit_activation(salience)
        # Expose novelty for neuromodulator controller
        signal.metadata["computed_novelty"] = self._last_novelty
        current = self._network_ctrl.current_mode

        # Creative mode trigger (Beaty 2018) — check before normal DMN/ECN switching
        if self._should_enter_creative(signal) and current != NetworkMode.CREATIVE:
            self._network_ctrl.switch_to(NetworkMode.CREATIVE, trigger="novel_high_importance")
            return Signal(
                type=SignalType.NETWORK_SWITCH,
                source=self.name,
                payload={"from": current.value, "to": "creative", "salience": salience},
            )

        if salience > ACTIVATION_THRESHOLD and current == NetworkMode.DMN:
            self._network_ctrl.switch_to(NetworkMode.ECN, trigger="salience_detected")
            return Signal(
                type=SignalType.NETWORK_SWITCH,
                source=self.name,
                payload={"from": "DMN", "to": "ECN", "salience": salience},
            )
        elif signal.type == SignalType.GWT_BROADCAST and signal.payload.get("status") == "task_complete":
            if current == NetworkMode.ECN:
                self._network_ctrl.switch_to(NetworkMode.DMN, trigger="task_complete")
                return Signal(
                    type=SignalType.NETWORK_SWITCH,
                    source=self.name,
                    payload={"from": "ECN", "to": "DMN"},
                )
        return None

    def _should_enter_creative(self, signal: Signal) -> bool:
        """Detect conditions for creative mode (Beaty 2018).

        Creative mode is triggered when:
        1. ACC has accumulated borderline errors (error_ratio > 0.5)
        2. No procedural cache match (novel situation)
        3. High arousal (important enough to warrant creative effort)
        """
        meta = signal.metadata
        # ACC borderline conflict (close to strategy switch but not there yet)
        error_ratio = meta.get("acc_error_ratio", 0.0)
        # No procedural match (novel situation)
        no_procedure = not meta.get("cached_procedure")
        # High importance
        arousal = 0.0
        if signal.emotional_tag:
            arousal = signal.emotional_tag.arousal
        return error_ratio > 0.5 and no_procedure and arousal > 0.3

    def _compute_salience(self, signal: Signal) -> float:
        arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0

        # Memory-based novelty assessment (Sridharan 2008):
        # Low retrieval score = high novelty (unfamiliar input)
        retrieved = signal.metadata.get("retrieved_memories", [])
        if retrieved:
            best_score = max(m.get("score", 0) for m in retrieved)
            novelty = 1.0 - min(1.0, best_score)
        else:
            # No retrieved memories available — assume novel
            novelty = 0.8 if signal.type in (SignalType.EXTERNAL_INPUT, SignalType.TEXT_INPUT, SignalType.IMAGE_INPUT, SignalType.AUDIO_INPUT) else 0.1

        self._last_novelty = novelty  # Store for pipeline access
        return arousal * 0.6 + novelty * 0.4
