"""Anterior Cingulate Cortex — Conflict monitoring and error detection.

Spec ref: Section 4.2 ACC (Botvinick, 2001; Holroyd & Coles, 2002).
- Monitors plans for contradictions and low confidence
- Accumulates prediction errors from Cerebellum
- Triggers strategy switch when error threshold exceeded
- Tracks per-tool error history for informed conflict detection
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

CONFLICT_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.7
PAST_FAILURE_LOOKBACK = 5
PAST_FAILURE_AVG_THRESHOLD = 0.5
MAX_ERROR_HISTORY = 100


class AnteriorCingulateCortex(BrainRegion):
    """Conflict monitoring, error detection, and strategy switching."""

    def __init__(self):
        super().__init__(name="acc", position=Vec3(0, 30, 25), lobe=Lobe.FRONTAL, hemisphere=Hemisphere.BILATERAL)
        self.error_accumulator: float = 0.0
        self.strategy_switch_threshold: float = 3.0
        self._error_history: list[dict] = []

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.PLAN:
            return self._evaluate_plan(signal)

        elif signal.type in (SignalType.ACTION_RESULT, SignalType.PREDICTION_ERROR):
            return self._accumulate_error(signal)

        return None

    # ── Plan evaluation (Botvinick conflict monitoring) ─────────────

    def _evaluate_plan(self, signal: Signal) -> Signal | None:
        actions = signal.payload.get("actions", [])
        if not actions:
            return None

        conflict_score = 0.0
        reasons: list[str] = []

        # Patience modulation (Doya 2002 5-HT) — higher patience raises the
        # conflict threshold so the agent tolerates more conflict before
        # requesting re-deliberation.
        nm = signal.metadata.get("neuromodulators", {})
        patience = nm.get("patience", 0.5)
        effective_threshold = CONFLICT_THRESHOLD * (0.5 + patience)

        # 1. Low confidence detection — uncertain actions need deliberation
        for action in actions:
            conf = action.get("confidence", 0.5)
            if conf < LOW_CONFIDENCE_THRESHOLD:
                conflict_score += LOW_CONFIDENCE_THRESHOLD - conf
                reasons.append(f"low_confidence:{action.get('tool', '?')}={conf:.2f}")

        # 2. High risk detection — risky actions trigger indirect pathway
        for action in actions:
            risk = action.get("risk", 0.0)
            if risk > HIGH_RISK_THRESHOLD:
                conflict_score += risk * 0.5
                reasons.append(f"high_risk:{action.get('tool', '?')}={risk:.2f}")

        # 3. Contradictory actions — duplicate tools on same target
        if len(actions) > 1:
            seen_tools: dict[str, int] = {}
            for a in actions:
                tool = a.get("tool", "")
                seen_tools[tool] = seen_tools.get(tool, 0) + 1
            for tool, count in seen_tools.items():
                if count > 1:
                    conflict_score += 0.4
                    reasons.append(f"duplicate_tool:{tool}")

        # 4. Past error patterns — tools with high failure history
        for action in actions:
            tool = action.get("tool", "")
            past = [e for e in self._error_history if e.get("tool") == tool]
            if len(past) >= 3:
                recent = past[-PAST_FAILURE_LOOKBACK:]
                avg_err = sum(e["error"] for e in recent) / len(recent)
                if avg_err > PAST_FAILURE_AVG_THRESHOLD:
                    conflict_score += 0.3
                    reasons.append(f"past_failures:{tool}(avg={avg_err:.2f})")

        # 5. Emotional risk — high arousal + negative valence = caution
        if signal.emotional_tag:
            if signal.emotional_tag.valence < -0.5 and signal.emotional_tag.arousal > 0.7:
                conflict_score += 0.2
                reasons.append("high_emotional_risk")

        # 6. Knowledge confidence assessment
        knowledge_confidences = signal.payload.get("knowledge_confidences", [])
        ambiguous_count = sum(1 for c in knowledge_confidences if c == "AMBIGUOUS")
        if ambiguous_count > 0:
            conflict_score += 0.15 * min(ambiguous_count, 3)
            reasons.append(f"{ambiguous_count} AMBIGUOUS knowledge edges")

        self.emit_activation(min(1.0, conflict_score))

        if conflict_score >= effective_threshold:
            return Signal(
                type=SignalType.CONFLICT_DETECTED,
                source=self.name,
                payload={
                    "conflict_score": conflict_score,
                    "reasons": reasons,
                    "request": "deliberate_more",
                },
            )
        return None

    # ── Error accumulation (Holroyd & Coles ERN) ────────────────────

    def _accumulate_error(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.PREDICTION_ERROR:
            error = float(signal.payload.get("error", 0.5))
            tool = signal.payload.get("tool", "unknown")
        else:
            # ACTION_RESULT
            expected = signal.payload.get("expected", 0)
            actual = signal.payload.get("actual", 0)
            tool = signal.payload.get("tool", "unknown")
            try:
                error = abs(float(expected) - float(actual))
            except (TypeError, ValueError):
                error = 1.0 if expected != actual else 0.0

        # Track error history per tool
        self._error_history.append({"tool": tool, "error": error})
        if len(self._error_history) > MAX_ERROR_HISTORY:
            self._error_history = self._error_history[-MAX_ERROR_HISTORY:]

        self.error_accumulator += error

        # Patience modulation (Doya 2002 5-HT) — higher patience raises
        # the effective strategy switch threshold, making ACC more tolerant
        nm = signal.metadata.get("neuromodulators", {})
        patience = nm.get("patience", 0.5)
        effective_switch_threshold = self.strategy_switch_threshold * (0.5 + patience)

        self.emit_activation(
            min(1.0, self.error_accumulator / self.strategy_switch_threshold)
        )

        if self.error_accumulator >= effective_switch_threshold:
            self.error_accumulator = 0.0
            return Signal(
                type=SignalType.STRATEGY_SWITCH,
                source=self.name,
                payload={"reason": "error accumulation exceeded threshold"},
            )
        return None
