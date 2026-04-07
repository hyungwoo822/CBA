# Brain Anatomical Reorganization Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize all brain regions by anatomical lobe and hemisphere. Add missing regions (visual cortex, auditory cortex, Wernicke, Broca, brainstem, VTA). Support multimodal input (text, image, audio).

**Architecture:** Each BrainRegion gets `lobe` and `hemisphere` fields. New regions are organized by anatomical location. Pipeline detects input modality and routes through appropriate cortical areas before convergence at PFC. Dashboard updated with lobe-based grouping.

**Tech Stack:** Python 3.11+, existing BrainRegion base class

**Excluded (per user):** Motor cortex, somatosensory cortex, olfactory cortex, spatial processing

---

## Anatomical Map

```
LEFT HEMISPHERE                          RIGHT HEMISPHERE
┌─────────────────────────┐              ┌─────────────────────────┐
│ FRONTAL LOBE            │              │ FRONTAL LOBE            │
│  PFC_L (logic,planning) │              │  PFC_R (holistic,creat) │
│  Broca (lang production)│              │                         │
│  ACC (medial, bilateral)│              │  ACC (medial, bilateral)│
└─────────────────────────┘              └─────────────────────────┘
┌─────────────────────────┐              ┌─────────────────────────┐
│ TEMPORAL LOBE           │              │ TEMPORAL LOBE           │
│  Wernicke (lang compreh)│              │  Auditory_R (prosody)   │
│  Auditory_L (speech)    │              │  Amygdala_R (fast emot) │
│  Amygdala_L (consc emot)│              │  Hippocampus_R (visual) │
│  Hippocampus_L (verbal) │              │                         │
└─────────────────────────┘              └─────────────────────────┘
┌─────────────────────────┐              ┌─────────────────────────┐
│ OCCIPITAL LOBE          │              │ OCCIPITAL LOBE          │
│  Visual_L (right field) │              │  Visual_R (left field)  │
└─────────────────────────┘              └─────────────────────────┘

              MIDLINE / SUBCORTICAL
     ┌──────────────────────────────────┐
     │ DIENCEPHALON (간뇌)              │
     │  Thalamus (sensory relay)        │
     │  Hypothalamus (homeostasis)      │
     ├──────────────────────────────────┤
     │ MIDBRAIN (중뇌)                  │
     │  VTA (dopamine source)           │
     │  Superior Colliculus (attention) │
     ├──────────────────────────────────┤
     │ BRAINSTEM (뇌간)                 │
     │  Locus Coeruleus (NE)            │
     │  Raphe Nuclei (5-HT)            │
     │  Reticular Formation (arousal)   │
     ├──────────────────────────────────┤
     │ BASAL GANGLIA (bilateral)        │
     │  Striatum (Go/NoGo)             │
     ├──────────────────────────────────┤
     │ CEREBELLUM (소뇌)                │
     │  Forward models + prediction     │
     ├──────────────────────────────────┤
     │ INSULAR CORTEX                   │
     │  Salience Network hub            │
     └──────────────────────────────────┘
```

## AI Agent Functional Mapping

| Brain Region | AI Agent Function |
|---|---|
| Visual Cortex (후두엽) | Image input preprocessing — feature extraction via vision model |
| Auditory Cortex L (좌측두엽) | Audio speech recognition — speech-to-text |
| Auditory Cortex R (우측두엽) | Emotional prosody analysis — tone/sentiment from audio |
| Wernicke (좌측두엽) | Text comprehension — semantic parsing, intent extraction |
| Broca (좌전두엽) | Language production — response formatting, style |
| PFC L (좌전두엽) | Logical/sequential reasoning — structured planning |
| PFC R (우전두엽) | Holistic/creative reasoning — divergent thinking |
| VTA (중뇌) | Dopamine dynamics — reward prediction error |
| Brainstem | Arousal regulation — session management, sleep-wake |
| Superior Colliculus (중뇌) | Attention orienting — modality selection for multimodal input |

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `brain_agent/regions/base.py` | Add `lobe`, `hemisphere` to BrainRegion |
| Create | `brain_agent/regions/visual_cortex.py` | Image input processing |
| Create | `brain_agent/regions/auditory_cortex.py` | Audio input processing (L=speech, R=prosody) |
| Create | `brain_agent/regions/wernicke.py` | Language comprehension |
| Create | `brain_agent/regions/broca.py` | Language production |
| Create | `brain_agent/regions/brainstem.py` | Arousal regulation, neuromod nuclei |
| Create | `brain_agent/regions/vta.py` | Dopamine source (VTA/SNc) |
| Modify | `brain_agent/regions/prefrontal.py` | Add hemisphere attribute (L=logic, R=creative) |
| Modify | `brain_agent/regions/amygdala.py` | Add hemisphere (L=conscious, R=fast) |
| Modify | `brain_agent/core/network_modes.py` | Update MODE_REGIONS with new regions |
| Modify | `brain_agent/core/router.py` | Add multimodal routing tables |
| Modify | `brain_agent/core/signals.py` | Add IMAGE_INPUT, AUDIO_INPUT signal types |
| Modify | `brain_agent/pipeline.py` | Multimodal input routing, new region calls |
| Modify | `dashboard/src/constants/brainRegions.ts` | Add new regions with positions/colors |
| Modify | `dashboard/src/stores/brainState.ts` | Add new region entries |

---

## Chunk 1: BrainRegion Base + Existing Region Annotation

### Task 1.1: Add lobe and hemisphere to BrainRegion

**Files:**
- Modify: `brain_agent/regions/base.py`
- Test: `tests/regions/test_base.py`

- [ ] **Step 1: Update BrainRegion base class**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from brain_agent.core.signals import Signal


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
    DIENCEPHALON = "diencephalon"     # 간뇌: thalamus, hypothalamus
    MIDBRAIN = "midbrain"             # 중뇌: VTA, superior colliculus
    BRAINSTEM = "brainstem"           # 뇌간: LC, raphe, reticular
    CEREBELLUM = "cerebellum"         # 소뇌
    SUBCORTICAL = "subcortical"       # 기저핵 등


class Hemisphere(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"           # Midline or both hemispheres


class BrainRegion(ABC):
    def __init__(
        self,
        name: str,
        position: Vec3 | None = None,
        lobe: Lobe = Lobe.SUBCORTICAL,
        hemisphere: Hemisphere = Hemisphere.BILATERAL,
    ):
        self.name = name
        self.position = position or Vec3(0, 0, 0)
        self.lobe = lobe
        self.hemisphere = hemisphere
        self.activation_level: float = 0.0
        self._events: list[dict] = []

    @abstractmethod
    async def process(self, signal: Signal) -> Signal | None: ...

    def emit_activation(self, level: float):
        self.activation_level = max(0.0, min(1.0, level))
        self._events.append({"region": self.name, "level": self.activation_level})
```

- [ ] **Step 2: Annotate ALL existing regions with correct lobe/hemisphere**

Each existing region gets its anatomically correct lobe and hemisphere:

```python
# prefrontal.py — PFC is frontal, bilateral (will split later)
super().__init__(name="prefrontal_cortex", position=Vec3(0, 60, 20),
                 lobe=Lobe.FRONTAL, hemisphere=Hemisphere.BILATERAL)

# acc.py — ACC is frontal (medial), bilateral
super().__init__(name="acc", position=Vec3(0, 30, 25),
                 lobe=Lobe.FRONTAL, hemisphere=Hemisphere.BILATERAL)

# amygdala.py — Amygdala is temporal (medial), bilateral
super().__init__(name="amygdala", position=Vec3(-30, -5, -20),
                 lobe=Lobe.TEMPORAL, hemisphere=Hemisphere.BILATERAL)

# basal_ganglia.py — Subcortical, bilateral
super().__init__(name="basal_ganglia", position=Vec3(-15, 0, 5),
                 lobe=Lobe.SUBCORTICAL, hemisphere=Hemisphere.BILATERAL)

# cerebellum.py — Cerebellum, bilateral
super().__init__(name="cerebellum", position=Vec3(0, -50, -30),
                 lobe=Lobe.CEREBELLUM, hemisphere=Hemisphere.BILATERAL)

# thalamus.py — Diencephalon, bilateral
super().__init__(name="thalamus", position=Vec3(0, 0, 0),
                 lobe=Lobe.DIENCEPHALON, hemisphere=Hemisphere.BILATERAL)

# hypothalamus.py — Diencephalon, bilateral
super().__init__(name="hypothalamus", position=Vec3(0, -10, -15),
                 lobe=Lobe.DIENCEPHALON, hemisphere=Hemisphere.BILATERAL)

# salience_network.py — Insular cortex, bilateral
super().__init__(name="salience_network", position=Vec3(30, 20, 10),
                 lobe=Lobe.INSULAR, hemisphere=Hemisphere.BILATERAL)
```

- [ ] **Step 3: Run tests, fix any broken tests**
- [ ] **Step 4: Commit**

---

## Chunk 2: New Cortical Regions (Visual, Auditory, Wernicke, Broca)

### Task 2.1: Visual Cortex (후두엽)

**Files:**
- Create: `brain_agent/regions/visual_cortex.py`
- Test: `tests/regions/test_visual_cortex.py`

```python
"""Visual Cortex — Image input processing.

Brain mapping: Occipital lobe (V1, V2, ventral stream).
Bilateral but functionally unified for image processing.

AI agent function: Preprocesses image inputs, extracts descriptions
and visual features. Routes visual information to PFC and hippocampus.
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class VisualCortex(BrainRegion):
    """Processes image inputs into structured visual descriptions."""

    def __init__(self):
        super().__init__(
            name="visual_cortex",
            position=Vec3(0, -40, -10),
            lobe=Lobe.OCCIPITAL,
            hemisphere=Hemisphere.BILATERAL,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type != SignalType.IMAGE_INPUT:
            return signal

        image_data = signal.payload.get("image_data")
        if not image_data:
            return signal

        # V1: Basic feature extraction (edges, shapes, colors)
        # V2: Pattern grouping
        # Ventral stream (what pathway): Object recognition
        # In AI agent: this would call a vision model or extract metadata
        signal.payload["visual_features"] = {
            "has_image": True,
            "description": signal.payload.get("image_description", ""),
        }
        signal.payload["modality"] = "visual"

        self.emit_activation(0.7)
        return signal
```

### Task 2.2: Auditory Cortex (측두엽)

**Files:**
- Create: `brain_agent/regions/auditory_cortex.py`
- Test: `tests/regions/test_auditory_cortex.py`

```python
"""Auditory Cortex — Audio input processing.

Brain mapping: Superior temporal gyrus.
  Left: Speech processing, phonological analysis (Heschl's gyrus L)
  Right: Prosody, emotional tone, music (Heschl's gyrus R)

AI agent function:
  L hemisphere: Speech-to-text, language extraction
  R hemisphere: Emotional tone analysis from audio
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class AuditoryCortexLeft(BrainRegion):
    """Left auditory cortex — speech/language processing."""

    def __init__(self):
        super().__init__(
            name="auditory_cortex_l",
            position=Vec3(-35, -10, 10),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type != SignalType.AUDIO_INPUT:
            return signal

        # Speech processing: extract text from audio
        # In real implementation: call speech-to-text model
        transcript = signal.payload.get("transcript", "")
        signal.payload["text"] = transcript
        signal.payload["modality"] = "auditory"

        self.emit_activation(0.6)
        return signal


class AuditoryCortexRight(BrainRegion):
    """Right auditory cortex — prosody and emotional tone."""

    def __init__(self):
        super().__init__(
            name="auditory_cortex_r",
            position=Vec3(35, -10, 10),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.RIGHT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type != SignalType.AUDIO_INPUT:
            return signal

        # Prosody analysis: emotional tone from audio
        # In real implementation: call emotion detection model
        tone = signal.payload.get("emotional_tone", "neutral")
        signal.payload["prosody"] = {
            "tone": tone,
            "stress_level": signal.payload.get("stress_level", 0.0),
        }

        self.emit_activation(0.5)
        return signal
```

### Task 2.3: Wernicke's Area (좌측두엽)

**Files:**
- Create: `brain_agent/regions/wernicke.py`
- Test: `tests/regions/test_wernicke.py`

```python
"""Wernicke's Area — Language comprehension.

Brain mapping: Left posterior superior temporal gyrus (BA 22).
Function: Semantic comprehension of language input.

AI agent function: Deep text understanding — extracts intent,
entities, semantic structure BEFORE passing to PFC for reasoning.
Enriches the signal with comprehension metadata.
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


INTENT_KEYWORDS = {
    "question": {"how", "what", "why", "when", "where", "who", "which", "?"},
    "command": {"run", "execute", "do", "create", "delete", "update", "fix", "build"},
    "inform": {"fyi", "note", "btw", "update:", "status:"},
}


class WernickeArea(BrainRegion):
    """Language comprehension — semantic parsing and intent extraction."""

    def __init__(self):
        super().__init__(
            name="wernicke",
            position=Vec3(-40, -20, 15),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        text = signal.payload.get("text", "")
        if not text:
            return signal

        # Semantic comprehension
        words = set(text.lower().split())
        intent = "statement"
        for intent_type, keywords in INTENT_KEYWORDS.items():
            if words & keywords:
                intent = intent_type
                break

        signal.payload["comprehension"] = {
            "intent": intent,
            "word_count": len(words),
            "complexity": min(1.0, len(words) / 50),  # Rough complexity estimate
        }

        self.emit_activation(0.5 + min(0.5, len(words) / 100))
        return signal
```

### Task 2.4: Broca's Area (좌전두엽)

**Files:**
- Create: `brain_agent/regions/broca.py`
- Test: `tests/regions/test_broca.py`

```python
"""Broca's Area — Language production.

Brain mapping: Left inferior frontal gyrus (BA 44, 45).
Function: Language production, syntax, articulation planning.

AI agent function: Post-processes PFC output for response formatting.
Handles output style, verbosity, and language structure.
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class BrocaArea(BrainRegion):
    """Language production — response formatting and style."""

    def __init__(self):
        super().__init__(
            name="broca",
            position=Vec3(-30, 40, 15),
            lobe=Lobe.FRONTAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type not in (SignalType.ACTION_SELECTED, SignalType.PLAN):
            return signal

        # Language production: format output
        action = signal.payload.get("action", {})
        text = action.get("args", {}).get("text", "")

        if text:
            # Apply production rules (could be extended with style preferences)
            signal.payload["formatted_response"] = text.strip()

        self.emit_activation(0.6)
        return signal
```

### Task 2.5: Brainstem (뇌간)

**Files:**
- Create: `brain_agent/regions/brainstem.py`
- Test: `tests/regions/test_brainstem.py`

```python
"""Brainstem — Arousal regulation and basic life functions.

Brain mapping: Medulla, pons, midbrain.
Contains neuromodulator nuclei:
  - Locus Coeruleus (pons): NE source
  - Raphe Nuclei (pons/medulla): 5-HT source
  - Reticular Formation: arousal/consciousness regulation

AI agent function: Manages arousal state (awake/drowsy/sleep analog).
Modulates processing depth based on arousal level.
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class Brainstem(BrainRegion):
    """Arousal regulation and consciousness state management."""

    def __init__(self):
        super().__init__(
            name="brainstem",
            position=Vec3(0, -30, -25),
            lobe=Lobe.BRAINSTEM,
            hemisphere=Hemisphere.BILATERAL,
        )
        self.arousal_state: str = "awake"  # awake | drowsy | sleep
        self._idle_count: int = 0

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.RESOURCE_STATUS:
            pending = signal.payload.get("pending_requests", 0)
            if pending == 0:
                self._idle_count += 1
            else:
                self._idle_count = 0

            # State transitions
            if self._idle_count > 10:
                self.arousal_state = "sleep"
                self.emit_activation(0.1)
            elif self._idle_count > 5:
                self.arousal_state = "drowsy"
                self.emit_activation(0.3)
            else:
                self.arousal_state = "awake"
                self.emit_activation(0.5)

            signal.payload["arousal_state"] = self.arousal_state
            return signal

        elif signal.type == SignalType.EXTERNAL_INPUT:
            # Any input → wake up (reticular activating system)
            self.arousal_state = "awake"
            self._idle_count = 0
            self.emit_activation(0.6)
            return signal

        return signal
```

### Task 2.6: VTA (중뇌)

**Files:**
- Create: `brain_agent/regions/vta.py`
- Test: `tests/regions/test_vta.py`

```python
"""Ventral Tegmental Area — Dopamine source.

Brain mapping: Midbrain, ventral tegmental area.
Function: Primary dopamine neuron cluster. Receives prediction error
signals and broadcasts DA to striatum (BG) and PFC.

AI agent function: Acts as the anatomical source of reward_signal
updates. Integrates with NeuromodulatorController.
"""
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class VTA(BrainRegion):
    """Ventral Tegmental Area — dopamine source for reward signaling."""

    def __init__(self):
        super().__init__(
            name="vta",
            position=Vec3(0, -20, -20),
            lobe=Lobe.MIDBRAIN,
            hemisphere=Hemisphere.BILATERAL,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.PREDICTION_ERROR:
            error = float(signal.payload.get("error", 0.0))
            # VTA fires proportionally to prediction error magnitude
            self.emit_activation(min(1.0, error * 1.5))
            # Pass through — NeuromodulatorController handles the actual DA update
            signal.metadata["vta_activation"] = self.activation_level
            return signal

        elif signal.type == SignalType.ACTION_RESULT:
            # Outcome evaluation also passes through VTA
            self.emit_activation(0.3)
            return signal

        return signal
```

- [ ] **Steps for all Task 2.x: Write tests, implement, verify, commit**

---

## Chunk 3: Signal Types + Pipeline Multimodal Routing

### Task 3.1: Add Multimodal Signal Types

**Files:**
- Modify: `brain_agent/core/signals.py`

Add:
```python
class SignalType(str, Enum):
    # ... existing types ...
    IMAGE_INPUT = "image_input"
    AUDIO_INPUT = "audio_input"
    MULTIMODAL_INPUT = "multimodal_input"
```

### Task 3.2: Update Routing Tables

**Files:**
- Modify: `brain_agent/core/router.py`

Add multimodal routes:
```python
ECN_ROUTES[SignalType.IMAGE_INPUT] = ["visual_cortex", "thalamus", "salience_network"]
ECN_ROUTES[SignalType.AUDIO_INPUT] = ["auditory_cortex_l", "auditory_cortex_r", "thalamus", "salience_network"]
```

### Task 3.3: Update MODE_REGIONS

**Files:**
- Modify: `brain_agent/core/network_modes.py`

```python
MODE_REGIONS = {
    NetworkMode.DMN: {"hippocampus", "consolidation"},
    NetworkMode.ECN: {
        "prefrontal_cortex", "acc", "basal_ganglia", "cerebellum",
        "thalamus", "wernicke", "broca",
        "visual_cortex", "auditory_cortex_l", "auditory_cortex_r",
    },
    NetworkMode.CREATIVE: {"prefrontal_cortex", "hippocampus", "acc"},
}

ALWAYS_ACTIVE: set[str] = {
    "amygdala", "hypothalamus", "salience_network",
    "brainstem", "vta",
}
```

### Task 3.4: Pipeline Multimodal Input Detection

**Files:**
- Modify: `brain_agent/pipeline.py`

In `process_request()`, detect input modality and route accordingly:

```python
async def process_request(
    self, text: str = "",
    image: bytes | None = None,
    audio: bytes | None = None,
) -> PipelineResult:
    # Determine input modality
    if image:
        input_signal = Signal(type=SignalType.IMAGE_INPUT, ...)
        # Route through visual cortex first
        input_signal = await self.visual_cortex.process(input_signal)
    elif audio:
        input_signal = Signal(type=SignalType.AUDIO_INPUT, ...)
        # Route through auditory cortices
        input_signal = await self.auditory_cortex_l.process(input_signal)
        input_signal = await self.auditory_cortex_r.process(input_signal)
    else:
        input_signal = Signal(type=SignalType.EXTERNAL_INPUT, ...)

    # Wernicke: comprehension (for all text-bearing signals)
    if input_signal.payload.get("text"):
        input_signal = await self.wernicke.process(input_signal)

    # Continue with existing pipeline: Thalamus → Amygdala → SN → ...
    # ...

    # Before returning response, route through Broca for formatting
    # ...
```

### Task 3.5: Register New Regions in Pipeline

Add to `ProcessingPipeline.__init__()`:
```python
# Cortical regions
self.visual_cortex = VisualCortex()
self.auditory_cortex_l = AuditoryCortexLeft()
self.auditory_cortex_r = AuditoryCortexRight()
self.wernicke = WernickeArea()
self.broca = BrocaArea()

# Subcortical
self.brainstem = Brainstem()
self.vta = VTA()
```

---

## Chunk 4: Dashboard Update

### Task 4.1: Update Dashboard Region Config

**Files:**
- Modify: `dashboard/src/constants/brainRegions.ts`
- Modify: `dashboard/src/stores/brainState.ts`

Add new regions with anatomically correct positions and lobe-based colors:

```typescript
// Lobe color scheme
// Frontal: blue family
// Temporal: cyan family
// Occipital: purple family
// Subcortical: orange/red family
// Brainstem/midbrain: pink family

export const REGION_CONFIG = {
  // ... existing regions ...
  visual_cortex: { position: [0, -10, -14], color: '#a855f7', scale: 1.0 },
  auditory_cortex_l: { position: [-12, -4, 8], color: '#06b6d4', scale: 0.6 },
  auditory_cortex_r: { position: [12, -4, 8], color: '#22d3ee', scale: 0.6 },
  wernicke: { position: [-11, -6, 5], color: '#0891b2', scale: 0.6 },
  broca: { position: [-8, 8, 10], color: '#2563eb', scale: 0.5 },
  brainstem: { position: [0, -8, -10], color: '#db2777', scale: 0.7 },
  vta: { position: [0, -6, -8], color: '#e11d48', scale: 0.4 },
}
```

---

## Execution Checklist

| Chunk | Tasks | Dependencies |
|-------|-------|-------------|
| 1: Base + Annotation | 1.1 (base class + annotate all existing) | None |
| 2: New Regions | 2.1-2.6 (visual, auditory, wernicke, broca, brainstem, VTA) | Chunk 1 |
| 3: Pipeline Integration | 3.1-3.5 (signals, routing, pipeline) | Chunk 1+2 |
| 4: Dashboard | 4.1 (constants, state) | Chunk 2 |

**Chunk 1 must be first.** Chunks 2 and 4 can partially overlap. Chunk 3 depends on 1+2.
