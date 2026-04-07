# Plan 2: Brain Region Processors + Network Implementation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the brain region processors (BrainRegion base + 8 regions), Thalamic Router, Global Workspace, Triple Network state machine, and neuromodulators — completing the core agent processing pipeline.

**Architecture:** GWT Orchestrator dispatches signals through Thalamic Router to specialist brain region processors. Only PFC calls LLM; all others are algorithm/rule-based. Triple Network (DMN/ECN/SN) state machine controls which regions are active.

**Tech Stack:** Python 3.11+, pytest, asyncio

**Spec:** `docs/superpowers/specs/2026-03-16-brain-agent-design.md` Sections 4-6

---

## File Structure

```
brain_agent/
├── core/
│   ├── neuromodulators.py      # Global params (urgency, learning_rate, patience, reward)
│   ├── network_modes.py        # Triple Network state machine (DMN/ECN/SN/CREATIVE)
│   ├── router.py               # Thalamic Router (routing table + TRN gating)
│   └── workspace.py            # Global Workspace (competition + broadcast)
├── regions/
│   ├── __init__.py
│   ├── base.py                 # BrainRegion ABC
│   ├── prefrontal.py           # PFC — LLM planning (stub, no real LLM yet)
│   ├── acc.py                  # ACC — conflict monitoring
│   ├── amygdala.py             # Amygdala — emotional tagging
│   ├── basal_ganglia.py        # Basal Ganglia — Go/NoGo gate
│   ├── cerebellum.py           # Cerebellum — prediction/error
│   ├── thalamus.py             # Thalamus — input preprocessing
│   ├── hypothalamus.py         # Hypothalamus — resource monitoring
│   └── salience_network.py     # SN — mode switching
├── pipeline.py                 # ProcessingPipeline — end-to-end request flow
tests/
├── core/
│   ├── test_neuromodulators.py
│   ├── test_network_modes.py
│   ├── test_router.py
│   └── test_workspace.py
├── regions/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_acc.py
│   ├── test_amygdala.py
│   ├── test_basal_ganglia.py
│   ├── test_cerebellum.py
│   └── test_salience_network.py
├── test_pipeline.py
```

---

## Chunk 1: Core Infrastructure (Neuromodulators + Network Modes)

### Task 1: Neuromodulators

**Files:** Create `brain_agent/core/neuromodulators.py`, `tests/core/test_neuromodulators.py`

```python
# tests/core/test_neuromodulators.py
from brain_agent.core.neuromodulators import Neuromodulators

def test_default_values():
    nm = Neuromodulators()
    assert nm.urgency == 0.5
    assert nm.learning_rate == 0.5
    assert nm.patience == 0.5
    assert nm.reward_signal == 0.0

def test_clamp_values():
    nm = Neuromodulators()
    nm.urgency = 1.5
    assert nm.urgency == 1.0
    nm.urgency = -0.5
    assert nm.urgency == 0.0

def test_update():
    nm = Neuromodulators()
    nm.update(urgency=0.8, patience=0.3)
    assert nm.urgency == 0.8
    assert nm.patience == 0.3

def test_snapshot():
    nm = Neuromodulators()
    nm.update(urgency=0.7)
    snap = nm.snapshot()
    assert snap["urgency"] == 0.7
    assert isinstance(snap, dict)
```

```python
# brain_agent/core/neuromodulators.py
"""Neuromodulatory global parameters. Spec ref: Section 5.6."""
from __future__ import annotations

class Neuromodulators:
    def __init__(self):
        self._urgency = 0.5
        self._learning_rate = 0.5
        self._patience = 0.5
        self._reward_signal = 0.0

    @property
    def urgency(self): return self._urgency
    @urgency.setter
    def urgency(self, v): self._urgency = max(0.0, min(1.0, v))

    @property
    def learning_rate(self): return self._learning_rate
    @learning_rate.setter
    def learning_rate(self, v): self._learning_rate = max(0.0, min(1.0, v))

    @property
    def patience(self): return self._patience
    @patience.setter
    def patience(self, v): self._patience = max(0.0, min(1.0, v))

    @property
    def reward_signal(self): return self._reward_signal
    @reward_signal.setter
    def reward_signal(self, v): self._reward_signal = max(-1.0, min(1.0, v))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def snapshot(self) -> dict:
        return {"urgency": self._urgency, "learning_rate": self._learning_rate,
                "patience": self._patience, "reward_signal": self._reward_signal}
```

### Task 2: Network Modes (Triple Network State Machine)

**Files:** Create `brain_agent/core/network_modes.py`, `tests/core/test_network_modes.py`

```python
# tests/core/test_network_modes.py
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController

def test_default_mode_is_dmn():
    ctrl = TripleNetworkController()
    assert ctrl.current_mode == NetworkMode.DMN

def test_switch_to_ecn():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    assert ctrl.current_mode == NetworkMode.ECN

def test_active_regions_differ_by_mode():
    ctrl = TripleNetworkController()
    dmn_regions = ctrl.get_active_regions()
    ctrl.switch_to(NetworkMode.ECN)
    ecn_regions = ctrl.get_active_regions()
    assert dmn_regions != ecn_regions

def test_ecn_regions():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    active = ctrl.get_active_regions()
    assert "prefrontal_cortex" in active
    assert "acc" in active
    assert "basal_ganglia" in active
    assert "cerebellum" in active

def test_dmn_regions():
    ctrl = TripleNetworkController()
    active = ctrl.get_active_regions()
    assert "hippocampus" in active
    assert "prefrontal_cortex" not in active

def test_switch_emits_history():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    ctrl.switch_to(NetworkMode.DMN)
    assert len(ctrl.switch_history) == 2
```

```python
# brain_agent/core/network_modes.py
"""Triple Network State Machine. Spec ref: Section 5.5.
DMN (idle/consolidation), ECN (task processing), CREATIVE (DMN+ECN coupling)."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field

class NetworkMode(str, Enum):
    DMN = "default_mode"
    ECN = "executive_control"
    CREATIVE = "creative"

MODE_REGIONS = {
    NetworkMode.DMN: {"hippocampus", "consolidation"},
    NetworkMode.ECN: {"prefrontal_cortex", "acc", "basal_ganglia", "cerebellum", "thalamus"},
    NetworkMode.CREATIVE: {"prefrontal_cortex", "hippocampus", "acc"},
}
ALWAYS_ACTIVE = {"amygdala", "hypothalamus", "salience_network"}

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
```

---

## Chunk 2: Thalamic Router + Global Workspace

### Task 3: Thalamic Router

**Files:** Create `brain_agent/core/router.py`, `tests/core/test_router.py`

```python
# tests/core/test_router.py
import pytest
from brain_agent.core.router import ThalamicRouter
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController
from brain_agent.core.neuromodulators import Neuromodulators

@pytest.fixture
def router():
    return ThalamicRouter(
        network_ctrl=TripleNetworkController(),
        neuromodulators=Neuromodulators(),
    )

def test_route_plan_in_ecn(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"content": "test"})
    targets = router.resolve_targets(sig)
    assert "acc" in targets
    assert "basal_ganglia" in targets

def test_plan_suppressed_in_dmn(router):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={})
    targets = router.resolve_targets(sig)
    assert len(targets) == 0

def test_broadcast_goes_to_all(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.GWT_BROADCAST, source="workspace", payload={})
    targets = router.resolve_targets(sig)
    assert len(targets) > 3

def test_priority_boosted_by_arousal(router):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={},
                 emotional_tag=EmotionalTag(valence=-0.5, arousal=0.9))
    p = router.compute_priority(sig)
    sig_neutral = Signal(type=SignalType.PLAN, source="pfc", payload={})
    p_neutral = router.compute_priority(sig_neutral)
    assert p > p_neutral

def test_conflict_forces_ecn_in_dmn(router):
    # DMN mode, conflict should force-route to PFC
    sig = Signal(type=SignalType.CONFLICT_DETECTED, source="acc", payload={})
    targets = router.resolve_targets(sig)
    assert "prefrontal_cortex" in targets

def test_routing_events_emitted(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={})
    router.resolve_targets(sig)
    assert len(router.event_log) == 1
```

```python
# brain_agent/core/router.py
"""Thalamic Router — intelligent message routing with gating.
Spec ref: Section 5.3. Routing table + TRN gating by network mode."""
from __future__ import annotations
from dataclasses import dataclass, field
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController
from brain_agent.core.neuromodulators import Neuromodulators

FORCE_ECN_TYPES = {SignalType.CONFLICT_DETECTED, SignalType.STRATEGY_SWITCH}

ECN_ROUTES: dict[SignalType, list[str]] = {
    SignalType.EXTERNAL_INPUT: ["thalamus", "salience_network"],
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
}
DMN_ROUTES: dict[SignalType, list[str]] = {
    SignalType.EXTERNAL_INPUT: ["thalamus", "salience_network"],
    SignalType.ACTION_RESULT: ["hippocampus", "amygdala"],
    SignalType.CONFLICT_DETECTED: ["prefrontal_cortex"],
    SignalType.STRATEGY_SWITCH: ["prefrontal_cortex"],
    SignalType.EMOTIONAL_TAG: ["hippocampus"],
    SignalType.CONSOLIDATION_TRIGGER: ["consolidation"],
    SignalType.RESOURCE_STATUS: ["hypothalamus"],
    SignalType.ENCODE: ["hippocampus"],
    SignalType.RETRIEVE: ["hippocampus"],
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
        elif mode == NetworkMode.ECN or mode == NetworkMode.CREATIVE:
            targets = ECN_ROUTES.get(signal.type, [])
        else:
            targets = DMN_ROUTES.get(signal.type, [])
        self.event_log.append(RoutingEvent(
            source=signal.source, targets=targets,
            signal_type=signal.type.value, priority=self.compute_priority(signal)))
        return targets

    def compute_priority(self, signal: Signal) -> float:
        base = signal.priority
        arousal_boost = 0.0
        if signal.emotional_tag:
            arousal_boost = signal.emotional_tag.arousal * 0.5
        return base * (1.0 + arousal_boost) * (0.5 + self._neuromod.urgency * 0.5)
```

### Task 4: Global Workspace

**Files:** Create `brain_agent/core/workspace.py`, `tests/core/test_workspace.py`

```python
# tests/core/test_workspace.py
import pytest
from brain_agent.core.workspace import GlobalWorkspace
from brain_agent.core.signals import Signal, SignalType, EmotionalTag

def test_submit_and_compete():
    gw = GlobalWorkspace()
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={"x": 1}),
              salience=0.8, goal_relevance=0.7)
    gw.submit(Signal(type=SignalType.EMOTIONAL_TAG, source="amygdala", payload={"x": 2}),
              salience=0.3, goal_relevance=0.2)
    winner = gw.compete()
    assert winner is not None
    assert winner.source == "pfc"

def test_no_winner_below_threshold():
    gw = GlobalWorkspace(ignition_threshold=0.9)
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={}),
              salience=0.1, goal_relevance=0.1)
    assert gw.compete() is None

def test_competition_clears_queue():
    gw = GlobalWorkspace()
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={}),
              salience=0.8, goal_relevance=0.8)
    gw.compete()
    assert gw.compete() is None  # queue empty

def test_high_arousal_boosts_score():
    gw = GlobalWorkspace()
    sig_calm = Signal(type=SignalType.PLAN, source="pfc", payload={},
                      emotional_tag=EmotionalTag(valence=0, arousal=0.1))
    sig_excited = Signal(type=SignalType.PLAN, source="pfc", payload={},
                         emotional_tag=EmotionalTag(valence=0, arousal=0.9))
    s1 = gw._compute_score(sig_calm, salience=0.5, goal_relevance=0.5)
    s2 = gw._compute_score(sig_excited, salience=0.5, goal_relevance=0.5)
    assert s2 > s1
```

```python
# brain_agent/core/workspace.py
"""Global Workspace — competition + broadcast. Spec ref: Section 5.4.
Baars (1988) / Dehaene (2011). Modules compete, winner broadcast to all."""
from __future__ import annotations
from dataclasses import dataclass
from brain_agent.core.signals import Signal

DEFAULT_IGNITION_THRESHOLD = 0.3

@dataclass
class _Candidate:
    signal: Signal
    salience: float
    goal_relevance: float
    score: float = 0.0

class GlobalWorkspace:
    def __init__(self, ignition_threshold: float = DEFAULT_IGNITION_THRESHOLD):
        self._threshold = ignition_threshold
        self._candidates: list[_Candidate] = []

    def submit(self, signal: Signal, salience: float = 0.5, goal_relevance: float = 0.5):
        score = self._compute_score(signal, salience, goal_relevance)
        self._candidates.append(_Candidate(signal=signal, salience=salience,
                                            goal_relevance=goal_relevance, score=score))

    def compete(self) -> Signal | None:
        if not self._candidates:
            return None
        self._candidates.sort(key=lambda c: (-c.score, c.signal.timestamp))
        winner = self._candidates[0]
        self._candidates.clear()
        if winner.score < self._threshold:
            return None
        return winner.signal

    def _compute_score(self, signal: Signal, salience: float, goal_relevance: float) -> float:
        arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
        return salience * 0.4 + arousal * 0.3 + goal_relevance * 0.3
```

---

## Chunk 3: Brain Region Processors

### Task 5: BrainRegion Base + Amygdala + ACC

**Files:** Create `brain_agent/regions/base.py`, `brain_agent/regions/amygdala.py`, `brain_agent/regions/acc.py` and their tests.

```python
# brain_agent/regions/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from brain_agent.core.signals import Signal

@dataclass
class Vec3:
    x: float; y: float; z: float

class BrainRegion(ABC):
    def __init__(self, name: str, position: Vec3 | None = None):
        self.name = name
        self.position = position or Vec3(0,0,0)
        self.activation_level: float = 0.0
        self._events: list[dict] = []

    @abstractmethod
    async def process(self, signal: Signal) -> Signal | None: ...

    def emit_activation(self, level: float):
        self.activation_level = max(0.0, min(1.0, level))
        self._events.append({"region": self.name, "level": self.activation_level})
```

Amygdala tests:
```python
# tests/regions/test_amygdala.py
import pytest
from brain_agent.regions.amygdala import Amygdala
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def amygdala():
    return Amygdala()

async def test_tags_normal_input(amygdala):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "please read file.py"})
    result = await amygdala.process(sig)
    assert result.emotional_tag is not None
    assert result.emotional_tag.arousal < 0.5

async def test_tags_error_with_high_arousal(amygdala):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "CRITICAL ERROR: server crashed"})
    result = await amygdala.process(sig)
    assert result.emotional_tag.arousal > 0.5
    assert result.emotional_tag.valence < 0

async def test_sets_priority_for_critical(amygdala):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "security breach detected"})
    result = await amygdala.process(sig)
    assert result.emotional_tag.arousal >= 0.7
```

ACC tests:
```python
# tests/regions/test_acc.py
import pytest
from brain_agent.regions.acc import AnteriorCingulateCortex
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def acc():
    return AnteriorCingulateCortex()

async def test_no_conflict_passes_through(acc):
    sig = Signal(type=SignalType.PLAN, source="pfc",
                 payload={"actions": [{"tool": "read_file"}]})
    result = await acc.process(sig)
    assert result is None or result.type != SignalType.CONFLICT_DETECTED

async def test_error_accumulation(acc):
    for _ in range(5):
        sig = Signal(type=SignalType.ACTION_RESULT, source="cerebellum",
                     payload={"expected": 1.0, "actual": 0.2})
        await acc.process(sig)
    assert acc.error_accumulator > 0

async def test_strategy_switch_on_threshold(acc):
    acc.strategy_switch_threshold = 0.5
    result = None
    for _ in range(10):
        sig = Signal(type=SignalType.ACTION_RESULT, source="cerebellum",
                     payload={"expected": 1.0, "actual": 0.0})
        result = await acc.process(sig)
        if result and result.type == SignalType.STRATEGY_SWITCH:
            break
    assert result is not None
    assert result.type == SignalType.STRATEGY_SWITCH
```

### Task 6: Basal Ganglia + Cerebellum + Salience Network

**Files:** Create `brain_agent/regions/basal_ganglia.py`, `brain_agent/regions/cerebellum.py`, `brain_agent/regions/salience_network.py` and tests.

Basal Ganglia tests:
```python
# tests/regions/test_basal_ganglia.py
import pytest
from brain_agent.regions.basal_ganglia import BasalGanglia
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def bg():
    return BasalGanglia()

async def test_go_for_confident_action(bg):
    sig = Signal(type=SignalType.PLAN, source="pfc",
                 payload={"actions": [{"tool": "read_file", "confidence": 0.9}]})
    result = await bg.process(sig)
    assert result is not None
    assert result.type == SignalType.ACTION_SELECTED

async def test_nogo_for_low_confidence(bg):
    sig = Signal(type=SignalType.PLAN, source="pfc",
                 payload={"actions": [{"tool": "delete_all", "confidence": 0.1, "risk": 0.9}]})
    result = await bg.process(sig)
    assert result is None or result.type != SignalType.ACTION_SELECTED
```

Cerebellum tests:
```python
# tests/regions/test_cerebellum.py
import pytest
from brain_agent.regions.cerebellum import Cerebellum
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def cere():
    return Cerebellum()

async def test_predict_outcome(cere):
    sig = Signal(type=SignalType.ACTION_SELECTED, source="basal_ganglia",
                 payload={"tool": "read_file", "args": {"path": "test.py"}})
    result = await cere.process(sig)
    assert result is not None
    assert "predicted_outcome" in result.payload

async def test_small_error_returns_adjustment(cere):
    sig = Signal(type=SignalType.ACTION_RESULT, source="executor",
                 payload={"predicted": "success", "actual": "success", "error": 0.02})
    result = await cere.process(sig)
    # Small error — no escalation
    assert result is None or result.type != SignalType.PREDICTION_ERROR

async def test_large_error_escalates(cere):
    sig = Signal(type=SignalType.ACTION_RESULT, source="executor",
                 payload={"predicted": "success", "actual": "failure", "error": 0.8})
    result = await cere.process(sig)
    assert result is not None
    assert result.type == SignalType.PREDICTION_ERROR
```

Salience Network tests:
```python
# tests/regions/test_salience_network.py
import pytest
from brain_agent.regions.salience_network import SalienceNetworkRegion
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController

@pytest.fixture
def sn():
    ctrl = TripleNetworkController()
    return SalienceNetworkRegion(network_ctrl=ctrl)

async def test_high_salience_switches_to_ecn(sn):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "urgent request"},
                 emotional_tag=EmotionalTag(valence=-0.5, arousal=0.8))
    result = await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.ECN

async def test_low_salience_stays_dmn(sn):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "ok"},
                 emotional_tag=EmotionalTag(valence=0, arousal=0.05))
    await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.DMN

async def test_task_complete_switches_to_dmn(sn):
    sn._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.GWT_BROADCAST, source="workspace",
                 payload={"status": "task_complete"})
    await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.DMN
```

---

## Chunk 4: Processing Pipeline + Integration

### Task 7: Thalamus + Hypothalamus + PFC Stub

Create remaining region stubs and the orchestration pipeline.

### Task 8: ProcessingPipeline + End-to-End Test

Wire everything together into a single `ProcessingPipeline` that follows the Section 6 processing flow.
