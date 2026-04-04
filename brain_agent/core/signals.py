from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalType(str, Enum):
    PLAN = "plan"
    ACTION_SELECTED = "action_selected"
    ACTION_RESULT = "action_result"
    ENCODE = "encode"
    RETRIEVE = "retrieve"
    EXTERNAL_INPUT = "external_input"
    PREDICTION_ERROR = "prediction_error"
    CONFLICT_DETECTED = "conflict_detected"
    STRATEGY_SWITCH = "strategy_switch"
    EMOTIONAL_TAG = "emotional_tag"
    GWT_BROADCAST = "gwt_broadcast"
    CONSOLIDATION_TRIGGER = "consolidation_trigger"
    NETWORK_SWITCH = "network_switch"
    RESOURCE_STATUS = "resource_status"
    IMAGE_INPUT = "image_input"
    AUDIO_INPUT = "audio_input"
    TEXT_INPUT = "text_input"
    SELF_REFERENCE = "self_reference"
    SOCIAL_COGNITION = "social_cognition"


@dataclass
class EmotionalTag:
    valence: float
    arousal: float

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

    @classmethod
    def neutral(cls) -> EmotionalTag:
        return cls(valence=0.0, arousal=0.0)


@dataclass
class Signal:
    type: SignalType
    source: str
    payload: dict
    targets: list[str] | None = None
    priority: float = 0.5
    emotional_tag: EmotionalTag | None = None
    interaction_id: int = 0
    session_id: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: uuid.UUID = field(default_factory=uuid.uuid4)
