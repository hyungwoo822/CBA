"""Cerebellum — Prediction, forward models, and motor learning.

Spec ref: Section 4.2 Cerebellum (Ito, 2008).
- Maintains forward models per action type (tool)
- Predicts action outcomes based on accumulated experience
- Computes prediction errors and updates internal models
- Small errors → fine-tune (fast cerebellar loop)
- Large errors → escalate to ACC (slow cortical loop)
"""
from __future__ import annotations

from dataclasses import dataclass

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

MINOR_ERROR_THRESHOLD = 0.3
MIN_DATA_FOR_PREDICTION = 3


@dataclass
class ForwardModel:
    """Learned prediction model for a specific action type (tool).

    Tracks success/failure counts and cumulative error to build
    a prediction of likely outcome.
    """
    success_count: int = 0
    fail_count: int = 0
    total_error: float = 0.0

    @property
    def total(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total if self.total > 0 else 0.5

    @property
    def avg_error(self) -> float:
        return self.total_error / self.total if self.total > 0 else 0.5

    @property
    def predicted_outcome(self) -> str:
        return "success" if self.success_rate > 0.5 else "failure"

    @property
    def confidence(self) -> float:
        """Higher with more data and more extreme success rate."""
        if self.total < MIN_DATA_FOR_PREDICTION:
            return 0.5
        data_weight = min(1.0, self.total / 20)
        extremity = abs(self.success_rate - 0.5) * 2
        return 0.5 + data_weight * extremity * 0.5


class Cerebellum(BrainRegion):
    """Prediction engine with learning forward models."""

    def __init__(self):
        super().__init__(name="cerebellum", position=Vec3(0, -50, -30), lobe=Lobe.CEREBELLUM, hemisphere=Hemisphere.BILATERAL)
        self._forward_models: dict[str, ForwardModel] = {}

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.ACTION_SELECTED:
            return self._predict(signal)
        elif signal.type == SignalType.ACTION_RESULT:
            return self._evaluate(signal)
        return None

    # ── Forward prediction (Ito, 2008) ──────────────────────────────

    def _predict(self, signal: Signal) -> Signal:
        """Predict outcome before execution using learned forward model."""
        action = signal.payload.get("action", {})
        tool = action.get("tool", signal.payload.get("tool", "unknown"))

        model = self._forward_models.get(tool)
        if model and model.total >= MIN_DATA_FOR_PREDICTION:
            predicted = model.predicted_outcome
            confidence = model.confidence
        else:
            # Insufficient data → default optimistic prediction
            predicted = "success"
            confidence = 0.5

        signal.payload["predicted_outcome"] = predicted
        signal.payload["prediction_confidence"] = confidence
        self.emit_activation(0.5)
        return signal

    # ── Error evaluation & model update ─────────────────────────────

    def _evaluate(self, signal: Signal) -> Signal | None:
        """Compare prediction vs actual and update forward model."""
        error = signal.payload.get("error", 0.0)
        tool = signal.payload.get("tool", "unknown")
        try:
            error = float(error)
        except (TypeError, ValueError):
            error = 0.5

        # Update forward model (cerebellar learning)
        model = self._forward_models.setdefault(tool, ForwardModel())
        model.total_error += error
        if error <= MINOR_ERROR_THRESHOLD:
            model.success_count += 1
        else:
            model.fail_count += 1

        # Small error → absorbed (fast cerebellar loop, no escalation)
        if error <= MINOR_ERROR_THRESHOLD:
            return None

        # Large error → escalate to ACC (slow cortical loop)
        self.emit_activation(min(1.0, error))
        return Signal(
            type=SignalType.PREDICTION_ERROR,
            source=self.name,
            payload={
                "error": error,
                "tool": tool,
                "predicted": signal.payload.get("predicted"),
                "actual": signal.payload.get("actual"),
            },
        )

    def get_model_stats(self) -> dict[str, dict]:
        """Return forward model statistics (for dashboard / debugging)."""
        return {
            tool: {
                "total": m.total,
                "success_rate": m.success_rate,
                "avg_error": m.avg_error,
                "confidence": m.confidence,
                "predicted_outcome": m.predicted_outcome,
            }
            for tool, m in self._forward_models.items()
        }
