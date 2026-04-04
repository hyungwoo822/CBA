"""Basal Ganglia — Action selection via Go/NoGo gating.

Spec ref: Section 4.2 Basal Ganglia (Mink, 1996; Frank, 2005).
- Default state: inhibit all actions
- Direct pathway (Go): reward expectation → release selected action
- Indirect pathway (NoGo): risk assessment → maintain inhibition
- Modulated by emotional state and neuromodulators
- Procedural memory success rate influences Go strength
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

GO_THRESHOLD = 0.3


class BasalGanglia(BrainRegion):
    """Action selection gate — only the best action passes through."""

    def __init__(self):
        super().__init__(name="basal_ganglia", position=Vec3(-15, 0, 5), lobe=Lobe.SUBCORTICAL, hemisphere=Hemisphere.BILATERAL)

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type != SignalType.PLAN:
            return None

        actions = signal.payload.get("actions", [])
        if not actions:
            return None

        emotional_tag = signal.emotional_tag
        neuromod = signal.metadata.get("neuromodulators", {})
        cached_proc = signal.metadata.get("cached_procedure")

        scored: list[tuple[dict, float]] = []
        for action in actions:
            # ── Direct pathway (Go) — reward expectation ────────────
            go_score = action.get("confidence", 0.5)

            # Procedural memory boost — familiar successful actions get a bonus
            if cached_proc and cached_proc.get("success_rate", 0) > 0.5:
                go_score *= 1.0 + cached_proc["success_rate"] * 0.3

            # Dopamine analogue — urgency speeds action selection
            urgency = neuromod.get("urgency", 0.5)
            go_score *= 1.0 + urgency * 0.2

            # Dopamine modulation (Schultz 1997) — reward expectation
            reward_signal = neuromod.get("reward_signal", 0.0)
            go_score *= 1.0 + reward_signal * 0.3  # Positive DA -> more Go, negative -> less

            # ── Indirect pathway (NoGo) — risk assessment ──────────
            nogo_score = action.get("risk", 0.2)

            # Patience modulation (Doya 2002 5-HT) — low patience = more
            # impulsive = less NoGo inhibition
            patience = neuromod.get("patience", 0.5)
            nogo_score += 0.15 * (1.0 - patience)

            # Emotional modulation (amygdala influence)
            if emotional_tag:
                # Negative valence increases caution (serotonin-like)
                if emotional_tag.valence < -0.3:
                    nogo_score += abs(emotional_tag.valence) * 0.15
                # High arousal can accelerate action (norepinephrine-like)
                if emotional_tag.arousal > 0.7:
                    go_score += emotional_tag.arousal * 0.1

            net = go_score - nogo_score
            scored.append((action, net))

        # ── Winner-take-all selection (Mink, 1996) ──────────────────
        winner = max(scored, key=lambda x: x[1])

        if winner[1] < GO_THRESHOLD:
            self.emit_activation(0.1)  # Low activation = all inhibited
            return None

        self.emit_activation(min(1.0, winner[1]))
        return Signal(
            type=SignalType.ACTION_SELECTED,
            source=self.name,
            payload={"action": winner[0], "go_score": winner[1]},
            emotional_tag=signal.emotional_tag,
        )
