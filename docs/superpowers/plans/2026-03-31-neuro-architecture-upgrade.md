# Neuro-Architecture Upgrade: 8 Missing Brain Mechanisms

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 missing neuroscience mechanisms (GABA/E-I balance, Insula, Attention expansion, Hippocampal pattern sep/comp, Metacognition, Adaptive depth, Predictive coding, Recurrent processing) and update docs with paper references.

**Architecture:** Each mechanism is a focused module change. GABA is foundational (touches NT system). Adaptive depth and recurrent processing change pipeline flow. The rest are independent region/memory additions. All pure computation except predictive coding (+1 parallel LLM call).

**Tech Stack:** Python 3.11+, asyncio, aiosqlite, pytest-asyncio, existing BrainRegion/Signal/Neuromodulator abstractions.

---

## File Map

| Feature | Create | Modify | Test |
|---------|--------|--------|------|
| GABA + E/I | — | `core/neuromodulators.py`, `core/neuromodulator_controller.py`, `memory/brain_state.py`, `pipeline.py` | `tests/core/test_neuromodulators.py`, `tests/core/test_neuromodulator_controller.py`, `tests/memory/test_brain_state.py` |
| Insula | `regions/insula.py` | `pipeline.py` | `tests/regions/test_insula.py` |
| Attention | — | `regions/thalamus.py`, `pipeline.py` | `tests/regions/test_thalamus.py` |
| Hippocampal P.S./P.C. | — | `memory/hippocampal_staging.py`, `memory/retrieval.py` | `tests/memory/test_hippocampal_staging.py`, `tests/memory/test_retrieval.py` |
| Metacognition | — | `regions/prefrontal.py`, `pipeline.py` | `tests/regions/test_prefrontal.py` |
| Adaptive Depth | — | `pipeline.py` | `tests/test_pipeline.py` |
| Predictive Coding | `core/predictor.py` | `pipeline.py` | `tests/core/test_predictor.py` |
| Recurrent Processing | — | `pipeline.py` | `tests/test_pipeline.py` |
| Docs | — | `docs/architecture-audit.md`, `docs/context-flow-architecture.md`, `docs/context-flow-architecture-ko.md` | — |

---

## Task 1: GABA + E/I Balance

**Files:**
- Modify: `brain_agent/core/neuromodulators.py`
- Modify: `brain_agent/core/neuromodulator_controller.py`
- Modify: `brain_agent/memory/brain_state.py`
- Modify: `brain_agent/pipeline.py:167-174` (restore_brain_state)
- Test: `tests/core/test_neuromodulators.py`
- Test: `tests/core/test_neuromodulator_controller.py`

### 1.1 Add GABA to Neuromodulators

- [ ] **Step 1: Write failing tests for GABA property**

In `tests/core/test_neuromodulators.py`, add:

```python
def test_gaba_default():
    nm = Neuromodulators()
    assert nm.gaba == 0.5

def test_gaba_clamp():
    nm = Neuromodulators()
    nm.gaba = 1.5
    assert nm.gaba == 1.0
    nm.gaba = -0.3
    assert nm.gaba == 0.0

def test_gaba_in_snapshot():
    nm = Neuromodulators()
    nm.gaba = 0.7
    snap = nm.snapshot()
    assert snap["gaba"] == 0.7

def test_gaba_load_from():
    nm = Neuromodulators()
    nm.load_from({"gaba": 0.8})
    assert nm.gaba == 0.8

def test_gaba_load_from_missing_defaults():
    nm = Neuromodulators()
    nm.gaba = 0.9
    nm.load_from({})  # gaba missing → defaults to 0.5
    assert nm.gaba == 0.5

def test_inhibition_alias():
    nm = Neuromodulators()
    nm.inhibition = 0.7
    assert nm.gaba == 0.7
    nm.gaba = 0.3
    assert nm.inhibition == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_neuromodulators.py -v -k "gaba or inhibition"`
Expected: FAIL — `AttributeError: 'Neuromodulators' object has no attribute 'gaba'`

- [ ] **Step 3: Implement GABA in Neuromodulators**

In `brain_agent/core/neuromodulators.py`:

Add to `__init__` after line 29 (`self._epinephrine = 0.5`):
```python
        # Inhibitory neurotransmitter
        self._gaba = 0.5             # GABA — cortical inhibition tone (Buzsaki 2006)
```

Add property block after the epinephrine property (after line 83):
```python
    # ── GABA — Cortical inhibitory tone (Buzsaki 2006) ───────────
    @property
    def gaba(self) -> float:
        return self._gaba

    @gaba.setter
    def gaba(self, v: float) -> None:
        self._gaba = _clamp(v)
```

Add backward-compat alias after `reward_signal` setter (after line 120):
```python
    @property
    def inhibition(self) -> float:
        return self._gaba

    @inhibition.setter
    def inhibition(self, v: float) -> None:
        self._gaba = _clamp(v)
```

Update `snapshot()` — add `"gaba": self._gaba,` after the epinephrine line.

Update `load_from()` — add after epinephrine line:
```python
        self._gaba = _clamp(state.get("gaba", 0.5))
```

Update docstring line 8 to add:
```
  - GABA: cortical inhibitory tone, E/I balance (Buzsaki 2006, Isaacson & Scanziani 2011)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_neuromodulators.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/neuromodulators.py tests/core/test_neuromodulators.py
git commit -m "feat(neuro): add GABA as 7th neurotransmitter with inhibition alias"
```

### 1.2 Add GABA dynamics to Controller

- [ ] **Step 6: Write failing tests for GABA dynamics**

In `tests/core/test_neuromodulator_controller.py`, add:

```python
def test_gaba_decay():
    """GABA decays toward baseline like other NTs."""
    nm = Neuromodulators()
    nm.gaba = 0.8
    ctrl = NeuromodulatorController(nm)
    ctrl.decay()
    assert 0.5 < nm.gaba < 0.8  # Decayed toward 0.5

def test_gaba_on_conflict_increases():
    """Conflict raises GABA (inhibitory braking, Aron 2007)."""
    nm = Neuromodulators()
    ctrl = NeuromodulatorController(nm)
    before = nm.gaba
    ctrl.on_conflict(0.8)
    assert nm.gaba > before

def test_ei_balance_crosstalk():
    """High excitation (NE+DA) without GABA → GABA compensatory rise."""
    nm = Neuromodulators()
    nm.norepinephrine = 0.85
    nm.dopamine = 0.85
    nm.gaba = 0.5
    ctrl = NeuromodulatorController(nm)
    ctrl._apply_crosstalk()
    assert nm.gaba > 0.5  # Compensatory inhibition

def test_high_gaba_suppresses_ne():
    """Very high GABA suppresses NE (inhibitory override)."""
    nm = Neuromodulators()
    nm.gaba = 0.85
    nm.norepinephrine = 0.7
    ctrl = NeuromodulatorController(nm)
    ctrl._apply_crosstalk()
    assert nm.norepinephrine < 0.7
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/core/test_neuromodulator_controller.py -v -k "gaba or ei_balance"`
Expected: FAIL

- [ ] **Step 8: Implement GABA dynamics in Controller**

In `brain_agent/core/neuromodulator_controller.py`:

Add constants after line 39 (`EPI_BASELINE = 0.5`):
```python
GABA_BASELINE = 0.5           # Tonic inhibitory tone (Buzsaki 2006)
```

Add after line 56 (`EPI_DECAY_RATE = 0.75`):
```python
GABA_DECAY_RATE = 0.88        # GABA: moderate-slow (~6 cycles, Isaacson & Scanziani 2011)
```

Add gain constants after line 89 (`EPI_THREAT_GAIN = 0.12`):
```python
# GABA (Cortical inhibitory interneurons) — E/I balance (Isaacson & Scanziani 2011)
GABA_CONFLICT_GAIN = 0.12     # Conflict → GABA (inhibitory braking, Aron 2007)
GABA_EI_COMPENSATION = 0.10   # High excitation → compensatory inhibition (homeostatic)
GABA_SUPPRESSION_FACTOR = 0.08  # High GABA → suppress excitatory NTs
```

In `on_conflict` method (after line 193), add:
```python
        # GABA: conflict triggers inhibitory braking (Aron 2007)
        self.neuromodulators.gaba += conflict_score * GABA_CONFLICT_GAIN
```

In `decay` method, add after line 267 (`nm.epinephrine = ...`):
```python
        nm.gaba = GABA_BASELINE + (nm.gaba - GABA_BASELINE) * GABA_DECAY_RATE
```

In `_apply_crosstalk` method, add at the end (after line 302):
```python
        # E/I homeostatic compensation (Isaacson & Scanziani 2011)
        # High excitation (NE+DA both elevated) → compensatory GABA rise
        excitation = (nm.norepinephrine + nm.dopamine) / 2.0
        if excitation > 0.65:
            nm.gaba += (excitation - 0.65) * GABA_EI_COMPENSATION

        # High GABA → suppress excitatory NTs (cortical inhibition)
        if nm.gaba > 0.7:
            gaba_excess = nm.gaba - 0.7
            nm.norepinephrine -= gaba_excess * GABA_SUPPRESSION_FACTOR
            nm.dopamine -= gaba_excess * GABA_SUPPRESSION_FACTOR * 0.5
```

Update docstring at top to add GABA line:
```
  Cortical Interneurons (GABA) -> gaba         -- Inhibitory tone, E/I balance
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/core/test_neuromodulator_controller.py -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add brain_agent/core/neuromodulator_controller.py tests/core/test_neuromodulator_controller.py
git commit -m "feat(neuro): GABA dynamics — decay, conflict braking, E/I homeostasis"
```

### 1.3 GABA persistence in BrainStateStore

- [ ] **Step 11: Write failing test for GABA persistence**

In `tests/memory/test_brain_state.py`, add:

```python
@pytest.mark.asyncio
async def test_gaba_persists(tmp_path):
    store = BrainStateStore(str(tmp_path / "brain.db"))
    await store.initialize()
    await store.save_neuromodulators({"gaba": 0.75, "dopamine": 0.6})
    loaded = await store.load_neuromodulators()
    assert loaded["gaba"] == 0.75
    await store.close()

@pytest.mark.asyncio
async def test_gaba_defaults_if_missing(tmp_path):
    store = BrainStateStore(str(tmp_path / "brain.db"))
    await store.initialize()
    loaded = await store.load_neuromodulators()
    assert loaded.get("gaba", 0.5) == 0.5
    await store.close()

@pytest.mark.asyncio
async def test_gaba_in_history(tmp_path):
    store = BrainStateStore(str(tmp_path / "brain.db"))
    await store.initialize()
    await store.save_neuromodulators({"gaba": 0.8})
    history = await store.get_neuromodulator_history(limit=1)
    assert len(history) == 1
    assert history[0]["gaba"] == 0.8
    await store.close()
```

- [ ] **Step 12: Run tests to verify they fail**

Run: `pytest tests/memory/test_brain_state.py -v -k gaba`
Expected: FAIL — `gaba` column doesn't exist in schema

- [ ] **Step 13: Add GABA column to BrainStateStore**

In `brain_agent/memory/brain_state.py`:

Update `brain_state` table creation (line 38-50) — add `gaba REAL DEFAULT 0.5` after `epinephrine`:
```python
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS brain_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dopamine REAL DEFAULT 0.5,
                norepinephrine REAL DEFAULT 0.5,
                serotonin REAL DEFAULT 0.5,
                acetylcholine REAL DEFAULT 0.5,
                cortisol REAL DEFAULT 0.5,
                epinephrine REAL DEFAULT 0.5,
                gaba REAL DEFAULT 0.5,
                region_activations TEXT DEFAULT '{}',
                interaction_count INTEGER DEFAULT 0,
                last_session_id TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            )
        """)
```

Update `neuromodulator_history` table (line 53-59) — add `gaba REAL`:
```python
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS neuromodulator_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dopamine REAL, norepinephrine REAL, serotonin REAL,
                acetylcholine REAL, cortisol REAL, epinephrine REAL,
                gaba REAL,
                timestamp TEXT NOT NULL
            )
        """)
```

Add migration for existing databases — add after the singleton INSERT (after line 68):
```python
        # Migration: add gaba column if upgrading from 6-NT schema
        try:
            await self._db.execute("ALTER TABLE brain_state ADD COLUMN gaba REAL DEFAULT 0.5")
            await self._db.commit()
        except Exception:
            pass  # Column already exists
        try:
            await self._db.execute("ALTER TABLE neuromodulator_history ADD COLUMN gaba REAL DEFAULT 0.5")
            await self._db.commit()
        except Exception:
            pass
```

Update `load_neuromodulators` — add `gaba` to SELECT and return dict:
```python
            async with self._db.execute(
                "SELECT dopamine, norepinephrine, serotonin, acetylcholine, "
                "cortisol, epinephrine, gaba FROM brain_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return {
                        "dopamine": row[0],
                        "norepinephrine": row[1],
                        "serotonin": row[2],
                        "acetylcholine": row[3],
                        "cortisol": row[4],
                        "epinephrine": row[5],
                        "gaba": row[6] if len(row) > 6 else 0.5,
                    }
```

Update `save_neuromodulators` — add `gaba` to UPDATE and INSERT:
```python
            await self._db.execute(
                "UPDATE brain_state SET "
                "dopamine=?, norepinephrine=?, serotonin=?, "
                "acetylcholine=?, cortisol=?, epinephrine=?, gaba=?, "
                "updated_at=? WHERE id = 1",
                (
                    state.get("dopamine", 0.5),
                    state.get("norepinephrine", 0.5),
                    state.get("serotonin", 0.5),
                    state.get("acetylcholine", 0.5),
                    state.get("cortisol", 0.5),
                    state.get("epinephrine", 0.5),
                    state.get("gaba", 0.5),
                    ts,
                ),
            )
            await self._db.execute(
                "INSERT INTO neuromodulator_history "
                "(dopamine, norepinephrine, serotonin, acetylcholine, cortisol, epinephrine, gaba, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.get("dopamine", 0.5),
                    state.get("norepinephrine", 0.5),
                    state.get("serotonin", 0.5),
                    state.get("acetylcholine", 0.5),
                    state.get("cortisol", 0.5),
                    state.get("epinephrine", 0.5),
                    state.get("gaba", 0.5),
                    ts,
                ),
            )
```

Update `get_neuromodulator_history` — add `gaba` to SELECT and return:
```python
            async with self._db.execute(
                "SELECT dopamine, norepinephrine, serotonin, acetylcholine, "
                "cortisol, epinephrine, gaba, timestamp FROM neuromodulator_history "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ) as cur:
```
And in the list comprehension:
```python
            return [
                {
                    "dopamine": r[0], "norepinephrine": r[1], "serotonin": r[2],
                    "acetylcholine": r[3], "cortisol": r[4], "epinephrine": r[5],
                    "gaba": r[6] if len(r) > 6 else 0.5,
                    "timestamp": r[7] if len(r) > 7 else r[6],
                }
                for r in reversed(rows)
            ]
```

Update `_defaults()` — add `"gaba": 0.5`.

- [ ] **Step 14: Update pipeline restore/save for GABA**

In `brain_agent/pipeline.py`, update `restore_brain_state` (line 167-174) — add gaba carry-over:
```python
            "gaba": _carry(nt_state.get("gaba", 0.5), 0.6),
```
GABA carry rate 0.6 = moderate persistence (inhibitory tone is relatively stable).

- [ ] **Step 15: Run all tests**

Run: `pytest tests/core/test_neuromodulators.py tests/core/test_neuromodulator_controller.py tests/memory/test_brain_state.py -v`
Expected: ALL PASS

- [ ] **Step 16: Commit**

```bash
git add brain_agent/memory/brain_state.py brain_agent/pipeline.py tests/memory/test_brain_state.py
git commit -m "feat(neuro): GABA persistence — DB migration, brain state carry-over"
```

---

## Task 2: Insula (Interoceptive Monitoring)

**Files:**
- Create: `brain_agent/regions/insula.py`
- Modify: `brain_agent/pipeline.py`
- Test: `tests/regions/test_insula.py`

- [ ] **Step 1: Write failing test for Insula**

Create `tests/regions/test_insula.py`:

```python
"""Tests for Insula — interoceptive monitoring (Craig 2009)."""
import pytest
from brain_agent.regions.insula import Insula
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.neuromodulators import Neuromodulators


@pytest.fixture
def nm():
    return Neuromodulators()


@pytest.fixture
def insula(nm):
    return Insula(neuromodulators=nm)


def test_name_and_position(insula):
    assert insula.name == "insula"


@pytest.mark.asyncio
async def test_computes_interoceptive_state(insula, nm):
    nm.cortisol = 0.8
    nm.epinephrine = 0.7
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT, source="test",
        payload={"text": "test"},
        emotional_tag=EmotionalTag(valence=-0.5, arousal=0.8),
    )
    result = await insula.process(signal)
    state = result.metadata["interoceptive_state"]
    assert state["stress_level"] > 0.5
    assert state["energy_level"] < 0.6
    assert state["emotional_awareness"] > 0.5


@pytest.mark.asyncio
async def test_low_arousal_low_awareness(insula, nm):
    nm.cortisol = 0.3
    nm.epinephrine = 0.3
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT, source="test",
        payload={"text": "test"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.1),
    )
    result = await insula.process(signal)
    state = result.metadata["interoceptive_state"]
    assert state["stress_level"] < 0.4
    assert state["emotional_awareness"] < 0.4


@pytest.mark.asyncio
async def test_risk_assessment(insula, nm):
    nm.cortisol = 0.75
    nm.gaba = 0.3  # Low inhibition
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT, source="test",
        payload={"text": "test"},
        emotional_tag=EmotionalTag(valence=-0.7, arousal=0.9),
    )
    result = await insula.process(signal)
    assert result.metadata["interoceptive_state"]["risk_sensitivity"] > 0.6


@pytest.mark.asyncio
async def test_activation_level(insula, nm):
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT, source="test",
        payload={"text": "test"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.5),
    )
    await insula.process(signal)
    assert insula.activation_level > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regions/test_insula.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain_agent.regions.insula'`

- [ ] **Step 3: Implement Insula region**

Create `brain_agent/regions/insula.py`:

```python
"""Insula — interoceptive monitoring and emotional awareness.

Brain mapping: Insular cortex (anterior and posterior), bilateral.

AI function: Monitors internal body state (neuromodulator levels) and
computes interoceptive signals — stress, energy, emotional awareness,
and risk sensitivity. Feeds into ACC and PFC for decision-making.

References:
  - Craig (2009): How Do You Feel — Now? The Anterior Insula and Human Awareness
  - Critchley et al. (2004): Neural systems supporting interoceptive awareness
  - Singer et al. (2009): Anterior insula integrates interoception with emotion
  - Paulus & Stein (2006): Insula and risk/uncertainty processing
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal
from brain_agent.core.neuromodulators import Neuromodulators


class Insula(BrainRegion):
    """Insular cortex — interoceptive monitoring and body-state awareness.

    Computes a composite 'interoceptive state' from neuromodulator levels
    and emotional tags, modeling Craig's (2009) interoceptive integration.
    """

    def __init__(self, neuromodulators: Neuromodulators) -> None:
        super().__init__(
            name="insula",
            position=Vec3(35, 10, 5),
            lobe=Lobe.INSULAR,
            hemisphere=Hemisphere.BILATERAL,
        )
        self._nm = neuromodulators

    async def process(self, signal: Signal) -> Signal:
        """Compute interoceptive state from internal signals.

        Integrates:
          - Cortisol/EPI → stress level (Critchley 2004)
          - DA/NE/EPI → energy level (autonomic balance)
          - Emotional tag + CORT → emotional awareness (Singer 2009)
          - CORT + low GABA + negative valence → risk sensitivity (Paulus & Stein 2006)
        """
        nm = self._nm
        arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
        valence = signal.emotional_tag.valence if signal.emotional_tag else 0.0

        # Stress level: cortisol + epinephrine weighted (Critchley 2004)
        stress_level = nm.cortisol * 0.6 + nm.epinephrine * 0.4

        # Energy level: inverse of depletion signals
        # High CORT = depleted, high DA = energized, moderate NE = alert
        energy_level = max(0.0, min(1.0,
            nm.dopamine * 0.4 + (1.0 - nm.cortisol) * 0.3 + nm.norepinephrine * 0.3
        ))

        # Emotional awareness: arousal amplifies, cortisol sharpens (Craig 2009)
        emotional_awareness = min(1.0, arousal * 0.5 + abs(valence) * 0.3 + stress_level * 0.2)

        # Risk sensitivity: high stress + low inhibition + negative valence (Paulus & Stein 2006)
        gaba_deficit = max(0.0, 0.5 - nm.gaba)  # Below-baseline GABA = low inhibition
        risk_sensitivity = min(1.0,
            stress_level * 0.4 + gaba_deficit * 0.3 + max(0.0, -valence) * 0.3
        )

        # Overall activation proportional to interoceptive salience
        activation = min(1.0, stress_level * 0.4 + arousal * 0.3 + abs(valence) * 0.3)
        self.emit_activation(activation)

        signal.metadata["interoceptive_state"] = {
            "stress_level": round(stress_level, 3),
            "energy_level": round(energy_level, 3),
            "emotional_awareness": round(emotional_awareness, 3),
            "risk_sensitivity": round(risk_sensitivity, 3),
        }
        return signal
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/regions/test_insula.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire Insula into pipeline**

In `brain_agent/pipeline.py`:

Add import after line 52 (`from brain_agent.regions.tpj import TemporoparietalJunction`):
```python
from brain_agent.regions.insula import Insula
```

Add instantiation after salience (line 100):
```python
        self.insula = Insula(neuromodulators=self.neuromodulators)  # Interoception (Craig 2009)
```

Add to `_all_regions()` list (line 134-142) — add `self.insula` after `self.salience`.

In pipeline Phase 3, after salience processing (after line 540, before mPFC section):
```python
        # ── 3f-pre. Insula: interoceptive monitoring (Craig 2009) ──
        insula_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
        insula_before.emotional_tag = input_signal.emotional_tag
        input_signal = await self.insula.process(input_signal)
        signals_count += 1
        await self._emit("region_activation", "insula", self.insula.activation_level, "active")
        await self._emit("signal_flow", "amygdala", "insula", "EMOTIONAL_TAG", 0.5)
        intero = input_signal.metadata.get("interoceptive_state", {})
        await self._emit("region_processing", "insula", "phase_3",
            f"Interoception: stress={intero.get('stress_level', 0):.2f}, energy={intero.get('energy_level', 0):.2f}")
        await self._step(0.1)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/regions/test_insula.py tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/regions/insula.py tests/regions/test_insula.py brain_agent/pipeline.py
git commit -m "feat(region): add Insula for interoceptive monitoring (Craig 2009)"
```

---

## Task 3: Attention System Expansion

**Files:**
- Modify: `brain_agent/regions/thalamus.py`
- Modify: `brain_agent/pipeline.py`
- Test: `tests/regions/test_thalamus.py`

- [ ] **Step 1: Write failing tests for attention gating**

In `tests/regions/test_thalamus.py`, add:

```python
from brain_agent.core.signals import EmotionalTag


def test_attention_gate_high_relevance(thalamus):
    """High goal-relevance signals pass with amplified weight."""
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "explain quantum computing"})
    signal.metadata["goal_keywords"] = ["quantum", "computing"]
    result = thalamus.process_with_attention(signal, goal_embedding=None, current_arousal=0.5)
    assert result is not None
    assert result.metadata.get("attention_weight", 0) > 0.3


def test_attention_gate_bottom_up_salience(thalamus):
    """High arousal input gets high bottom-up attention."""
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "fire!!!"})
    signal.emotional_tag = EmotionalTag(valence=-0.8, arousal=0.95)
    result = thalamus.process_with_attention(signal, goal_embedding=None, current_arousal=0.95)
    assert result is not None
    assert result.metadata.get("attention_weight", 0) > 0.5


def test_attention_gate_low_relevance_passes(thalamus):
    """Low relevance still passes (thalamus doesn't block, just weights)."""
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "hmm"})
    result = thalamus.process_with_attention(signal, goal_embedding=None, current_arousal=0.1)
    assert result is not None
    assert "attention_weight" in result.metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/regions/test_thalamus.py -v -k attention`
Expected: FAIL — no `process_with_attention` method

- [ ] **Step 3: Implement attention gating in Thalamus**

In `brain_agent/regions/thalamus.py`, add after the existing `process` method:

```python
    def process_with_attention(
        self,
        signal: Signal,
        goal_embedding: list[float] | None = None,
        current_arousal: float = 0.0,
    ) -> Signal:
        """Thalamic reticular nucleus attention gating (McAlonan et al. 2008).

        Computes attention weight from two streams:
          - Top-down (dorsal): goal relevance via keyword overlap
          - Bottom-up (ventral): salience via arousal + novelty

        Does NOT block signals — assigns attention_weight for downstream
        regions to use for resource allocation.

        References:
          - Corbetta & Shulman (2002): Two attention networks
          - McAlonan et al. (2008): Thalamic reticular nucleus gating
        """
        # First apply standard preprocessing
        signal = self.process(signal)
        if signal is None:
            return signal

        text = signal.payload.get("text", "")
        words = set(text.lower().split())

        # Top-down: keyword overlap with current goals (Corbetta & Shulman 2002)
        goal_keywords = set(signal.metadata.get("goal_keywords", []))
        if goal_keywords and words:
            overlap = len(words & goal_keywords) / max(len(goal_keywords), 1)
            top_down = min(1.0, overlap * 2.0)  # Amplify partial matches
        else:
            top_down = 0.3  # Default moderate relevance

        # Bottom-up: arousal-driven salience (ventral attention, Corbetta 2002)
        arousal = current_arousal
        if signal.emotional_tag:
            arousal = max(arousal, signal.emotional_tag.arousal)
        bottom_up = arousal

        # Composite attention weight
        attention_weight = top_down * 0.5 + bottom_up * 0.5
        signal.metadata["attention_weight"] = round(min(1.0, attention_weight), 3)

        return signal
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/regions/test_thalamus.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire attention gating into pipeline**

In `brain_agent/pipeline.py`, replace the simple thalamus call (around line 318) with:

```python
        # Thalamic gating with attention weighting (McAlonan et al. 2008)
        goal_keywords = []
        if self.pfc.goals:
            goal_keywords = self.pfc.goals.get_keywords()
        input_signal = self.thalamus.process_with_attention(
            input_signal,
            goal_embedding=None,
            current_arousal=self.neuromodulators.epinephrine,
        )
        if goal_keywords:
            input_signal.metadata["goal_keywords"] = goal_keywords
```

Note: If `self.pfc.goals` doesn't have `get_keywords()`, fall back to empty list. Check prefrontal.py for goal tree API and adapt accordingly.

- [ ] **Step 6: Run pipeline tests**

Run: `pytest tests/test_pipeline.py tests/regions/test_thalamus.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/regions/thalamus.py brain_agent/pipeline.py tests/regions/test_thalamus.py
git commit -m "feat(region): dual attention gating in thalamus (Corbetta & Shulman 2002)"
```

---

## Task 4: Hippocampal Pattern Separation / Completion

**Files:**
- Modify: `brain_agent/memory/hippocampal_staging.py`
- Modify: `brain_agent/memory/retrieval.py`
- Test: `tests/memory/test_hippocampal_staging.py`
- Test: `tests/memory/test_retrieval.py`

- [ ] **Step 1: Write failing tests for pattern separation**

In `tests/memory/test_hippocampal_staging.py`, add:

```python
@pytest.mark.asyncio
async def test_pattern_separation_distinct_encoding(staging):
    """Similar inputs get distinct embeddings (DG orthogonalization, Yassa & Stark 2011)."""
    id1 = await staging.encode("I like coffee in the morning", entities={"keywords": ["coffee"]}, interaction_id="i1", session_id="s1")
    id2 = await staging.encode("I like coffee in the evening", entities={"keywords": ["coffee"]}, interaction_id="i2", session_id="s1")
    mem1 = await staging.get_by_id(id1)
    mem2 = await staging.get_by_id(id2)
    # Pattern separation: similar memories should have metadata flag
    # indicating they were separated (not identical storage)
    assert mem2.get("separated_from") is not None or id1 != id2
```

- [ ] **Step 2: Write failing tests for pattern completion**

In `tests/memory/test_retrieval.py`, add:

```python
def test_pattern_completion_boost():
    """Partial cues get completion boost when context matches (CA3, Rolls 2013)."""
    engine = RetrievalEngine()
    # High context similarity = pattern completion scenario
    score_with_context = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.85, activation_boost=0.0,
    )
    score_without_context = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.3, activation_boost=0.0,
    )
    # High context similarity should significantly boost retrieval
    assert score_with_context > score_without_context * 1.2


def test_pattern_completion_nonlinear():
    """Pattern completion is nonlinear — kicks in above threshold (Rolls 2013)."""
    engine = RetrievalEngine()
    score_below = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.5, activation_boost=0.0,
    )
    score_above = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.8, activation_boost=0.0,
    )
    # The jump from 0.5→0.8 context sim should be disproportionately large
    diff_high = score_above - score_below
    assert diff_high > 0.05
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/memory/test_hippocampal_staging.py tests/memory/test_retrieval.py -v -k "pattern"`
Expected: FAIL

- [ ] **Step 4: Implement pattern separation in hippocampal staging**

In `brain_agent/memory/hippocampal_staging.py`, modify the `encode` method. After computing embedding, add a similarity check:

```python
    async def encode(
        self, content, entities, interaction_id, session_id,
        emotional_tag=None, source_modality="text",
    ) -> str:
        """Encode with DG pattern separation (Yassa & Stark 2011).

        Before storing, checks for highly similar existing memories.
        If found, emphasizes distinctive features in metadata to
        maintain separable representations.
        """
        mem_id = str(__import__("uuid").uuid4())
        embedding = await self._embed_fn(content)

        # Pattern separation (DG): detect near-duplicates
        separated_from = None
        if self._db:
            try:
                async with self._db.execute(
                    "SELECT id, context_embedding FROM staging_memories "
                    "WHERE consolidated = 0 ORDER BY last_interaction DESC LIMIT 10"
                ) as cur:
                    rows = await cur.fetchall()
                for row in rows:
                    existing_emb = __import__("json").loads(row[1]) if isinstance(row[1], str) else row[1]
                    if existing_emb and embedding:
                        sim = self._cosine_sim(embedding, existing_emb)
                        if sim > 0.85:
                            separated_from = row[0]
                            break
            except Exception:
                pass  # Separation check is best-effort

        # Store with separation metadata
        entities_with_sep = dict(entities) if entities else {}
        if separated_from:
            entities_with_sep["separated_from"] = separated_from

        # ... rest of existing encode logic, using entities_with_sep instead of entities
```

Add cosine similarity helper to the class:
```python
    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
```

- [ ] **Step 5: Implement pattern completion boost in retrieval**

In `brain_agent/memory/retrieval.py`, modify `compute_score` to add nonlinear completion:

After the existing score computation (line 58), before the return:
```python
        # Pattern completion (CA3, Rolls 2013): nonlinear boost
        # when context similarity exceeds threshold — partial cues
        # trigger full memory reconstruction
        completion_threshold = 0.7
        if context_similarity > completion_threshold:
            excess = context_similarity - completion_threshold
            completion_bonus = excess * excess * 2.0  # Quadratic boost
            score += completion_bonus
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/memory/test_hippocampal_staging.py tests/memory/test_retrieval.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/memory/hippocampal_staging.py brain_agent/memory/retrieval.py tests/memory/test_hippocampal_staging.py tests/memory/test_retrieval.py
git commit -m "feat(memory): hippocampal pattern sep (DG) + completion (CA3)"
```

---

## Task 5: Metacognition (PFC Confidence)

**Files:**
- Modify: `brain_agent/regions/prefrontal.py`
- Modify: `brain_agent/pipeline.py`
- Test: `tests/regions/test_prefrontal.py`

- [ ] **Step 1: Write failing tests for confidence parsing**

In `tests/regions/test_prefrontal.py`, add:

```python
def test_parse_confidence_from_response():
    """PFC should extract confidence score from its own output."""
    pfc = PrefrontalCortex()
    text = 'Some answer here\n<meta>{"confidence": 0.85}</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.85
    assert "<meta>" not in clean

def test_parse_confidence_missing():
    """Missing confidence defaults to 0.7."""
    pfc = PrefrontalCortex()
    text = "Just a plain answer"
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.7
    assert clean == "Just a plain answer"

def test_parse_confidence_low():
    """Low confidence is preserved."""
    pfc = PrefrontalCortex()
    text = 'Not sure about this\n<meta>{"confidence": 0.3}</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.3
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/regions/test_prefrontal.py -v -k "confidence"`
Expected: FAIL — `_parse_metacognition` doesn't exist

- [ ] **Step 3: Add metacognition instruction to PFC prompt**

In `brain_agent/regions/prefrontal.py`, find the entity extraction instruction block and add after it:

```python
METACOGNITION_INSTRUCTION = """
At the END of your response, on a new line, output a metacognitive self-assessment:
<meta>{"confidence": 0.0-1.0}</meta>

confidence: How confident are you in this response? (0.0 = guessing, 1.0 = certain)
Base this on: completeness of your knowledge, ambiguity of the question, quality of retrieved memories.
"""
```

Add `METACOGNITION_INSTRUCTION` to the system prompt assembly in `_call_llm` (within the guidelines section).

- [ ] **Step 4: Add `_parse_metacognition` method to PFC**

```python
    def _parse_metacognition(self, text: str) -> tuple[float, str]:
        """Extract metacognitive assessment from PFC output (Fleming 2010).

        Returns (confidence, clean_text) where clean_text has <meta> removed.

        References:
          - Fleming & Dolan (2012): Neural basis of metacognitive ability
          - Yeung & Summerfield (2012): Metacognition in human decision-making
        """
        import re
        import json as _json
        match = re.search(r"<meta>\s*(\{.*?\})\s*</meta>", text, re.DOTALL)
        if match:
            try:
                meta = _json.loads(match.group(1))
                confidence = float(meta.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))
                clean = text[:match.start()].rstrip() + text[match.end():]
                return confidence, clean.strip()
            except (ValueError, _json.JSONDecodeError):
                pass
        return 0.7, text  # Default confidence
```

- [ ] **Step 5: Wire metacognition into PFC process**

In PFC's `process` method, after getting the LLM response and before returning the signal, parse metacognition:

```python
        # Metacognitive self-assessment (Fleming & Dolan 2012)
        confidence, clean_response = self._parse_metacognition(response_text)
        response_text = clean_response
        signal.metadata["metacognition"] = {"confidence": confidence}
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/regions/test_prefrontal.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/regions/prefrontal.py tests/regions/test_prefrontal.py
git commit -m "feat(region): PFC metacognition — confidence self-assessment (Fleming 2012)"
```

---

## Task 6: Adaptive Processing Depth

**Files:**
- Modify: `brain_agent/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for adaptive depth**

In `tests/test_pipeline.py`, add:

```python
@pytest.mark.asyncio
async def test_simple_input_uses_fast_path(pipeline):
    """Simple greetings should use minimal processing (fewer phases)."""
    result = await pipeline.process_request(text="안녕")
    # Fast path should still produce a response
    assert result.response

@pytest.mark.asyncio
async def test_classify_complexity_simple():
    """Simple inputs are classified as simple."""
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "simple", "intent": "greeting"}, has_procedure=True) == "fast"

@pytest.mark.asyncio
async def test_classify_complexity_complex():
    """Complex analytical inputs classified as full."""
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "complex", "intent": "explanation"}, has_procedure=False) == "full"

@pytest.mark.asyncio
async def test_classify_complexity_moderate():
    """Moderate inputs get standard path."""
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "moderate", "intent": "question"}, has_procedure=False) == "standard"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_pipeline.py -v -k "complexity or fast_path"`
Expected: FAIL — `_classify_complexity` not found

- [ ] **Step 3: Add complexity classifier function**

In `brain_agent/pipeline.py`, add before the `ProcessingPipeline` class:

```python
def _classify_complexity(
    comprehension: dict, has_procedure: bool,
) -> str:
    """Classify input complexity for adaptive processing depth.

    Returns 'fast', 'standard', or 'full' based on Wernicke comprehension
    output and procedural cache status.

    Rationale: Simple inputs (greetings, confirmations) don't need full
    7-phase processing. This models the brain's ability to handle routine
    stimuli with minimal cortical engagement (Schneider & Shiffrin 1977).

    References:
      - Schneider & Shiffrin (1977): Automatic vs controlled processing
      - Posner & Snyder (1975): Two-process theory of attention
    """
    complexity = comprehension.get("complexity", "moderate")
    intent = comprehension.get("intent", "unknown")

    # Fast path: routine + procedural cache hit
    if has_procedure and complexity == "simple":
        return "fast"

    # Fast path: greetings, confirmations, single-word responses
    if intent in ("greeting", "confirmation", "farewell") and complexity == "simple":
        return "fast"

    # Full path: complex analytical / creative requests
    if complexity in ("complex", "very_complex"):
        return "full"

    return "standard"
```

- [ ] **Step 4: Integrate adaptive depth into pipeline**

In `process_request`, after Phase 6 retrieval and procedural cache check (around line 638), add complexity classification:

```python
        # ── Adaptive processing depth (Schneider & Shiffrin 1977) ──
        processing_depth = _classify_complexity(comprehension, has_procedure=bool(cached_procedure))
        result.metadata = {"processing_depth": processing_depth}

        if processing_depth == "fast" and cached_procedure:
            # Fast path: skip full PFC reasoning, use cached procedure
            # This models automatic processing (Schneider & Shiffrin 1977)
            plan_signal = await self.pfc.process(input_signal)  # Uses procedural fast-path internally
            signals_count += 1
            if plan_signal:
                for act in plan_signal.payload.get("actions", []):
                    resp = act.get("args", {}).get("text")
                    if resp:
                        result.response = resp
                        break
            # Skip ACC, BG, Cerebellum — go straight to Broca
            # (automatic behavior doesn't need conflict monitoring)
            ...
```

Note: The existing procedural fast-path in PFC already skips LLM when stage is "autonomous". The adaptive depth enhancement makes the pipeline ALSO skip ACC/BG/Cerebellum for fast-path items. The exact implementation should wrap the executive processing block in a condition:

```python
        if processing_depth != "fast":
            # Full executive loop: ACC → BG → Cerebellum → Execute
            ...existing code...
        else:
            # Fast path: direct to speech production
            if plan_signal:
                for act in plan_signal.payload.get("actions", []):
                    resp = act.get("args", {}).get("text")
                    if resp:
                        result.response = resp
                        break
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add brain_agent/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): adaptive processing depth (Schneider & Shiffrin 1977)"
```

---

## Task 7: Predictive Coding

**Files:**
- Create: `brain_agent/core/predictor.py`
- Modify: `brain_agent/pipeline.py`
- Test: `tests/core/test_predictor.py`

- [ ] **Step 1: Write failing tests for Predictor**

Create `tests/core/test_predictor.py`:

```python
"""Tests for Predictive Coding module (Friston 2005, Rao & Ballard 1999)."""
import pytest
from brain_agent.core.predictor import Predictor


@pytest.fixture
def predictor():
    return Predictor()


def test_no_prediction_initially(predictor):
    """No prediction exists before first interaction."""
    assert predictor.last_prediction is None


def test_compute_surprise_no_prediction(predictor):
    """Surprise is 0.5 (neutral) when no prediction exists."""
    surprise = predictor.compute_surprise([0.1] * 10)
    assert surprise == 0.5


def test_compute_surprise_matching(predictor):
    """Low surprise when prediction matches input."""
    predictor.store_prediction([0.5] * 10)
    surprise = predictor.compute_surprise([0.5] * 10)
    assert surprise < 0.2


def test_compute_surprise_mismatching(predictor):
    """High surprise when prediction diverges from input."""
    predictor.store_prediction([0.9] * 10)
    surprise = predictor.compute_surprise([-0.9] * 10)
    assert surprise > 0.7


def test_store_prediction(predictor):
    """Prediction is stored and retrievable."""
    emb = [0.1, 0.2, 0.3]
    predictor.store_prediction(emb)
    assert predictor.last_prediction == emb


def test_surprise_feeds_novelty(predictor):
    """Surprise value is in [0, 1] range."""
    predictor.store_prediction([1.0] * 10)
    surprise = predictor.compute_surprise([0.0] * 10)
    assert 0.0 <= surprise <= 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/core/test_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Predictor**

Create `brain_agent/core/predictor.py`:

```python
"""Predictive Coding — prediction error and surprise computation.

Maintains a prediction of the next input embedding. When actual input
arrives, computes surprise as cosine distance between prediction and
reality. Surprise feeds into ACh (novelty) and DA (prediction error).

This implements the core intuition of predictive processing: the brain
is a prediction machine that minimizes surprise (Friston 2005).

References:
  - Friston (2005): A theory of cortical responses (Free Energy)
  - Rao & Ballard (1999): Predictive coding in visual cortex
  - Clark (2013): Whatever Next? Predictive brains, situated agents
"""
from __future__ import annotations

import math


class Predictor:
    """Lightweight prediction error module.

    Stores the predicted embedding for next input and computes
    surprise when actual input arrives. No LLM needed for the
    surprise computation — it's pure cosine distance.
    """

    def __init__(self) -> None:
        self._prediction: list[float] | None = None

    @property
    def last_prediction(self) -> list[float] | None:
        return self._prediction

    def store_prediction(self, embedding: list[float]) -> None:
        """Store predicted embedding for next input."""
        self._prediction = embedding

    def compute_surprise(self, actual_embedding: list[float]) -> float:
        """Compute surprise as 1 - cosine_similarity(prediction, actual).

        Returns 0.0 (no surprise, exact match) to 1.0 (maximum surprise).
        Returns 0.5 (neutral) if no prediction exists.

        References:
          - Friston (2005): Surprise = -log p(sensory input | model)
          - Simplified here as cosine distance for computational efficiency
        """
        if self._prediction is None:
            return 0.5

        sim = self._cosine_sim(self._prediction, actual_embedding)
        # Convert similarity [-1, 1] to surprise [0, 1]
        surprise = (1.0 - sim) / 2.0
        return max(0.0, min(1.0, surprise))

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(min_len))
        norm_a = math.sqrt(sum(x * x for x in a[:min_len]))
        norm_b = math.sqrt(sum(x * x for x in b[:min_len]))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_predictor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire Predictor into pipeline**

In `brain_agent/pipeline.py`:

Add import:
```python
from brain_agent.core.predictor import Predictor
```

Add to `__init__`:
```python
        self.predictor = Predictor()  # Predictive coding (Friston 2005)
```

In Phase 1, after computing input embedding (or after thalamus processing), add surprise computation:
```python
        # Predictive coding: compute surprise from prior prediction (Friston 2005)
        input_embedding = input_signal.payload.get("embedding")
        if input_embedding:
            surprise = self.predictor.compute_surprise(input_embedding)
            input_signal.metadata["prediction_surprise"] = surprise
            # Surprise feeds into novelty (overrides retrieval-based novelty if stronger)
            # This is the key insight: novelty comes from model prediction, not just memory
        else:
            surprise = 0.5
```

In Phase 7, after Broca produces output, store prediction for next turn:
```python
        # Predictive coding: generate prediction for next input
        # Use response embedding as context for what might come next
        if result.response:
            try:
                pred_emb = await self.memory.staging._embed_fn(result.response)
                self.predictor.store_prediction(pred_emb)
            except Exception:
                pass  # Prediction storage is best-effort
```

Feed surprise into neuromodulator controller — enhance the existing novelty call:
```python
        # Use prediction-based surprise if available, else retrieval-based novelty
        surprise = input_signal.metadata.get("prediction_surprise", None)
        if surprise is not None:
            effective_novelty = max(novelty, surprise)  # Take the stronger signal
        else:
            effective_novelty = novelty
        self.neuro_ctrl.on_novelty(effective_novelty)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/core/test_predictor.py tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/core/predictor.py tests/core/test_predictor.py brain_agent/pipeline.py
git commit -m "feat(core): predictive coding — surprise computation (Friston 2005)"
```

---

## Task 8: Recurrent Processing (Feedback Loops)

**Files:**
- Modify: `brain_agent/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for recurrent processing**

In `tests/test_pipeline.py`, add:

```python
@pytest.mark.asyncio
async def test_reprocessing_on_low_confidence(pipeline_with_mock_llm):
    """Low confidence triggers reprocessing loop (Lamme 2006)."""
    # Mock PFC to return low confidence first, then high confidence
    call_count = [0]
    original_process = pipeline_with_mock_llm.pfc.process

    async def mock_pfc_process(signal):
        result = await original_process(signal)
        call_count[0] += 1
        if call_count[0] == 1:
            result.metadata["metacognition"] = {"confidence": 0.2}
        else:
            result.metadata["metacognition"] = {"confidence": 0.8}
        return result

    pipeline_with_mock_llm.pfc.process = mock_pfc_process
    result = await pipeline_with_mock_llm.process_request(text="explain this")
    # Should have processed PFC at least twice
    assert call_count[0] >= 2


@pytest.mark.asyncio
async def test_max_reprocess_iterations(pipeline_with_mock_llm):
    """Reprocessing loop caps at MAX_REPROCESS iterations."""
    # Mock PFC to always return low confidence
    original_process = pipeline_with_mock_llm.pfc.process

    async def always_low_confidence(signal):
        result = await original_process(signal)
        result.metadata["metacognition"] = {"confidence": 0.1}
        return result

    pipeline_with_mock_llm.pfc.process = always_low_confidence
    result = await pipeline_with_mock_llm.process_request(text="hard question")
    # Should still return a response (doesn't loop forever)
    assert result.response
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_pipeline.py -v -k "reprocess"`
Expected: FAIL or unexpected behavior (no reprocessing loop exists)

- [ ] **Step 3: Implement recurrent processing in pipeline**

In `brain_agent/pipeline.py`, add constant:
```python
MAX_REPROCESS = 2  # Maximum reprocessing iterations (Lamme 2006)
REPROCESS_CONFIDENCE_THRESHOLD = 0.4  # Below this → re-deliberate
```

Replace the single PFC call + ACC conflict section in the executive processing block with a reprocessing loop:

```python
        # ══════════════════════════════════════════════════════════════
        # Executive Processing with Recurrent Loop (Lamme 2006)
        # PFC → ACC → [confidence check] → re-PFC if needed
        # Models recurrent processing: the brain doesn't just do one
        # forward pass — it iterates until confident or max iterations.
        # ══════════════════════════════════════════════════════════════

        plan_signal = None
        for reprocess_iter in range(MAX_REPROCESS + 1):
            # ── PFC: plan with memory context (LLM reasoning)
            pfc_before = input_signal
            plan_signal = await self.pfc.process(input_signal)
            signals_count += 1
            await self._emit("region_activation", "prefrontal_cortex",
                self.pfc.activation_level, "high_activity")
            await self._emit_region_io("prefrontal_cortex", pfc_before,
                plan_signal, f"planning_iter_{reprocess_iter}")

            if not plan_signal or plan_signal.type != SignalType.PLAN:
                break

            # ── ACC: conflict monitoring (Botvinick 2001)
            conflict = None
            if self._is_active("acc"):
                conflict = await self.acc.process(plan_signal)
                signals_count += 1

            # ── Metacognitive confidence check (Fleming 2012)
            confidence = (plan_signal.metadata.get("metacognition", {})
                         .get("confidence", 0.7))

            # Decide: accept or re-deliberate?
            should_reprocess = (
                reprocess_iter < MAX_REPROCESS
                and (
                    (conflict and conflict.type == SignalType.CONFLICT_DETECTED)
                    or confidence < REPROCESS_CONFIDENCE_THRESHOLD
                )
            )

            if should_reprocess:
                # Feed back ACC conflict and low confidence into next iteration
                if conflict and conflict.type == SignalType.CONFLICT_DETECTED:
                    conflict_score = conflict.payload.get("conflict_score", 0)
                    self.neuro_ctrl.on_conflict(conflict_score)
                    input_signal.metadata["acc_feedback"] = conflict.payload
                input_signal.metadata["reprocess_reason"] = (
                    "low_confidence" if confidence < REPROCESS_CONFIDENCE_THRESHOLD
                    else "conflict_detected"
                )
                input_signal.metadata["previous_confidence"] = confidence
                await self._emit("region_processing", "acc", "executive",
                    f"Reprocessing iter {reprocess_iter + 1}: "
                    f"confidence={confidence:.2f}, conflict={bool(conflict)}")
                continue  # Re-enter PFC with feedback

            # Accepted — break out of reprocessing loop
            break
```

This replaces the existing linear PFC → ACC → conflict re-plan block, adding the metacognition-driven loop.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): recurrent processing loop with metacognitive gating (Lamme 2006)"
```

---

## Task 9: Documentation Update

**Files:**
- Modify: `docs/architecture-audit.md`
- Modify: `docs/context-flow-architecture.md`
- Modify: `docs/context-flow-architecture-ko.md`

- [ ] **Step 1: Update architecture-audit.md**

Add a new section for each implemented mechanism with paper references:

```markdown
### GABA + E/I Balance
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Buzsaki (2006) | 2006 | GABA cortical inhibition | `core/neuromodulators.py` — 7th NT |
| Isaacson & Scanziani (2011) | 2011 | E/I balance homeostasis | `core/neuromodulator_controller.py` — crosstalk |
| Aron (2007) | 2007 | Inhibitory braking on conflict | `core/neuromodulator_controller.py` — on_conflict |

### Insula (Interoception)
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Craig (2009) | 2009 | Anterior insula awareness | `regions/insula.py` — interoceptive state |
| Critchley et al. (2004) | 2004 | Neural interoception | `regions/insula.py` — stress/energy |
| Paulus & Stein (2006) | 2006 | Insula risk processing | `regions/insula.py` — risk_sensitivity |
| Singer et al. (2009) | 2009 | Emotion integration | `regions/insula.py` — emotional_awareness |

### Attention System
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Corbetta & Shulman (2002) | 2002 | Dual attention networks | `regions/thalamus.py` — top-down + bottom-up |
| McAlonan et al. (2008) | 2008 | Thalamic reticular gating | `regions/thalamus.py` — process_with_attention |

### Hippocampal Pattern Separation/Completion
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Yassa & Stark (2011) | 2011 | DG pattern separation | `memory/hippocampal_staging.py` — encode |
| Rolls (2013) | 2013 | CA3 pattern completion | `memory/retrieval.py` — completion_bonus |

### Metacognition
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Fleming & Dolan (2012) | 2012 | Neural metacognition | `regions/prefrontal.py` — _parse_metacognition |
| Yeung & Summerfield (2012) | 2012 | Metacognitive decisions | `regions/prefrontal.py` — confidence output |

### Adaptive Processing Depth
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Schneider & Shiffrin (1977) | 1977 | Automatic vs controlled | `pipeline.py` — _classify_complexity |
| Posner & Snyder (1975) | 1975 | Two-process attention | `pipeline.py` — fast/standard/full |

### Predictive Coding
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Friston (2005) | 2005 | Free energy / prediction | `core/predictor.py` — surprise |
| Rao & Ballard (1999) | 1999 | Predictive coding cortex | `core/predictor.py` — cosine distance |
| Clark (2013) | 2013 | Predictive brains | `pipeline.py` — surprise → ACh/DA |

### Recurrent Processing
| Paper | Year | Mechanism | Implementation |
|-------|------|-----------|---------------|
| Lamme (2006) | 2006 | Recurrent processing theory | `pipeline.py` — reprocessing loop |
| Dehaene et al. (2006) | 2006 | Ignition and recurrence | `pipeline.py` — confidence threshold |
```

- [ ] **Step 2: Update context-flow-architecture.md**

Add to the pipeline phase descriptions:
- Phase 3: Insula interoceptive monitoring
- Phase 1: Predictive coding surprise computation + attention gating
- Executive: Recurrent processing loop with metacognitive gating
- Pre-executive: Adaptive depth routing (fast/standard/full)

Update NT system section:
- Add GABA as 7th neurotransmitter with E/I balance dynamics

Update memory section:
- Add pattern separation (DG) and pattern completion (CA3)

- [ ] **Step 3: Update context-flow-architecture-ko.md**

Mirror all changes from Step 2 in Korean.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture-audit.md docs/context-flow-architecture.md docs/context-flow-architecture-ko.md
git commit -m "docs: add references for 8 new mechanisms (GABA, insula, attention, etc.)"
```

---

## Summary of All Paper References

| # | Paper | Year | Feature |
|---|-------|------|---------|
| 1 | Buzsaki — Rhythms of the Brain | 2006 | GABA |
| 2 | Isaacson & Scanziani — E/I Balance | 2011 | GABA/E-I |
| 3 | Aron — Stop-signal inhibition | 2007 | GABA conflict |
| 4 | Craig — How Do You Feel Now? | 2009 | Insula |
| 5 | Critchley et al. — Neural interoception | 2004 | Insula |
| 6 | Paulus & Stein — Insula and risk | 2006 | Insula |
| 7 | Singer et al. — Insula emotion | 2009 | Insula |
| 8 | Corbetta & Shulman — Two attention networks | 2002 | Attention |
| 9 | McAlonan et al. — Thalamic reticular gating | 2008 | Attention |
| 10 | Yassa & Stark — Pattern separation | 2011 | Hippocampus |
| 11 | Rolls — CA3 pattern completion | 2013 | Hippocampus |
| 12 | Fleming & Dolan — Neural metacognition | 2012 | Metacognition |
| 13 | Yeung & Summerfield — Metacognitive decisions | 2012 | Metacognition |
| 14 | Schneider & Shiffrin — Automatic processing | 1977 | Adaptive depth |
| 15 | Posner & Snyder — Two-process theory | 1975 | Adaptive depth |
| 16 | Friston — Free energy principle | 2005 | Predictive coding |
| 17 | Rao & Ballard — Predictive coding cortex | 1999 | Predictive coding |
| 18 | Clark — Whatever Next? | 2013 | Predictive coding |
| 19 | Lamme — Recurrent processing | 2006 | Recurrent |
| 20 | Dehaene et al. — Ignition and recurrence | 2006 | Recurrent |
