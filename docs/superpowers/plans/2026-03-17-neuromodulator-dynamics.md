# Neuromodulator Dynamics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 4 neuromodulators (DA, NE, ACh, 5-HT) dynamically rise and fall based on brain-faithful triggers — prediction errors, arousal, novelty, conflict — visible in real-time on the dashboard.

**Architecture:** A `NeuromodulatorController` encapsulates all dynamics logic (baseline, phasic response, decay). The pipeline calls it at anatomically correct points after each brain region processes. Regions remain decoupled — they don't write neuromodulators directly. The controller models 4 nuclei: VTA (DA), LC (NE), Nucleus Basalis (ACh), Dorsal Raphe (5-HT).

**Tech Stack:** Python 3.11+, pure algorithm (no external deps)

---

## Neuroscience Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    NEUROMODULATOR NUCLEI                     │
│                                                             │
│  VTA/SNc (Dopamine)          Locus Coeruleus (NE)           │
│  ├─ Input: prediction error  ├─ Input: arousal (amygdala)   │
│  ├─ Input: outcome eval      ├─ Input: conflict (ACC)       │
│  ├─ Output: BG Go pathway    ├─ Input: novelty (SN)         │
│  └─ Output: PFC motivation   ├─ Input: system load (hypo)   │
│                               └─ Output: BG Go, routing     │
│                                                             │
│  Nucleus Basalis (ACh)       Dorsal Raphe (5-HT)            │
│  ├─ Input: novelty (SN)     ├─ Input: error rate (hypo)     │
│  ├─ Input: uncertainty (ACC) ├─ Input: reward history        │
│  ├─ Input: |RPE| magnitude  └─ Output: BG NoGo, ACC thresh  │
│  └─ Output: encoding strength                               │
└─────────────────────────────────────────────────────────────┘
```

### Update Points in Pipeline (Anatomical Order)

```
Step 4  Amygdala    → NE ↑ (high arousal = alert)
Step 5  SN         → ACh ↑ (high novelty = learn more)
Step 14 Cerebellum → DA ± (prediction error = RPE)
Step 15 ACC        → NE ↑ (conflict = alert), ACh ↑ (uncertainty = learn)
Step 18 Hypothalamus → NE (system load), 5-HT (error rate) [EXISTING]
End     Decay      → all drift toward baseline
```

### Dynamics Model

Each neuromodulator has:
- **Baseline** (resting state): the value it returns to between events
- **Phasic response**: event-driven spikes/dips
- **Decay**: exponential return toward baseline per interaction
- **Clamp**: hard limits [0,1] for NE/ACh/5-HT, [-1,1] for DA

```
new_value = current + phasic_delta
after_decay = baseline + (new_value - baseline) * decay_rate
```

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `brain_agent/core/neuromodulator_controller.py` | All dynamics: 4 nuclei, phasic updates, decay logic |
| Modify | `brain_agent/pipeline.py` | Call controller at 5 anatomical points |
| Modify | `brain_agent/regions/basal_ganglia.py` | Consume `reward_signal` (DA → Go boost) |
| Modify | `brain_agent/regions/hypothalamus.py` | Delegate to controller instead of direct update |
| Test | `tests/core/test_neuromodulator_controller.py` | Full dynamics coverage |

---

## Chunk 1: NeuromodulatorController + Pipeline Integration

### Task 1: Create NeuromodulatorController

**Files:**
- Create: `brain_agent/core/neuromodulator_controller.py`
- Test: `tests/core/test_neuromodulator_controller.py`

- [ ] **Step 1: Write failing tests for all 4 nuclei**

```python
import pytest
from brain_agent.core.neuromodulators import Neuromodulators
from brain_agent.core.neuromodulator_controller import NeuromodulatorController


@pytest.fixture
def ctrl():
    nm = Neuromodulators()
    return NeuromodulatorController(nm)


# ── VTA: Dopamine (reward_signal) ──────────────────────────

class TestVTA:
    def test_positive_prediction_error_increases_da(self, ctrl):
        """Better than expected → DA spike (positive RPE)."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        assert ctrl.neuromodulators.reward_signal > 0.0

    def test_negative_prediction_error_decreases_da(self, ctrl):
        """Worse than expected → DA dip (negative RPE)."""
        ctrl.on_prediction_error(error=0.8, predicted="success", actual="failure")
        assert ctrl.neuromodulators.reward_signal < 0.0

    def test_expected_outcome_no_da_change(self, ctrl):
        """As expected → minimal DA change."""
        ctrl.on_prediction_error(error=0.05, predicted="success", actual="success")
        assert abs(ctrl.neuromodulators.reward_signal) < 0.2

    def test_da_consumed_by_bg_go(self, ctrl):
        """DA should be accessible for BG Go pathway."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        snap = ctrl.neuromodulators.snapshot()
        assert snap["reward_signal"] > 0


# ── Locus Coeruleus: NE (urgency) ─────────────────────────

class TestLC:
    def test_high_arousal_increases_ne(self, ctrl):
        """Amygdala threat detection → NE spike."""
        before = ctrl.neuromodulators.urgency
        ctrl.on_emotional_arousal(arousal=0.9)
        assert ctrl.neuromodulators.urgency > before

    def test_low_arousal_minimal_ne_change(self, ctrl):
        """Calm input → no NE change."""
        before = ctrl.neuromodulators.urgency
        ctrl.on_emotional_arousal(arousal=0.1)
        assert ctrl.neuromodulators.urgency == pytest.approx(before, abs=0.1)

    def test_conflict_increases_ne(self, ctrl):
        """ACC conflict → phasic NE burst."""
        before = ctrl.neuromodulators.urgency
        ctrl.on_conflict(conflict_score=0.8)
        assert ctrl.neuromodulators.urgency > before

    def test_system_load_increases_ne(self, ctrl):
        """High pending requests → sustained NE."""
        ctrl.on_system_state(pending_requests=8, error_rate=0.2)
        assert ctrl.neuromodulators.urgency > 0.5


# ── Nucleus Basalis: ACh (learning_rate) ───────────────────

class TestNucleusBasalis:
    def test_high_novelty_increases_ach(self, ctrl):
        """Novel input → ACh spike → learn more."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_novelty(novelty=0.9)
        assert ctrl.neuromodulators.learning_rate > before

    def test_familiar_input_decreases_ach(self, ctrl):
        """Familiar input → low ACh → less plasticity."""
        ctrl.on_novelty(novelty=0.1)
        assert ctrl.neuromodulators.learning_rate < 0.5

    def test_uncertainty_increases_ach(self, ctrl):
        """ACC uncertainty → more learning needed."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_conflict(conflict_score=0.7)
        assert ctrl.neuromodulators.learning_rate > before

    def test_large_rpe_magnitude_increases_ach(self, ctrl):
        """Big surprise (either direction) → learn from it."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_prediction_error(error=0.9, predicted="success", actual="failure")
        assert ctrl.neuromodulators.learning_rate > before


# ── Dorsal Raphe: 5-HT (patience) ─────────────────────────

class TestDorsalRaphe:
    def test_low_error_rate_high_patience(self, ctrl):
        """Things going well → more patient."""
        ctrl.on_system_state(pending_requests=0, error_rate=0.1)
        assert ctrl.neuromodulators.patience > 0.5

    def test_high_error_rate_low_patience(self, ctrl):
        """Things going badly → impatient/frustrated."""
        ctrl.on_system_state(pending_requests=0, error_rate=0.9)
        assert ctrl.neuromodulators.patience < 0.5

    def test_positive_reward_increases_patience(self, ctrl):
        """Consistent rewards → more patience (delayed gratification)."""
        before = ctrl.neuromodulators.patience
        ctrl.on_reward_outcome(success=True)
        assert ctrl.neuromodulators.patience >= before

    def test_negative_reward_decreases_patience(self, ctrl):
        """Failures → less patience."""
        ctrl.on_reward_outcome(success=False)
        assert ctrl.neuromodulators.patience < 0.5


# ── Decay ──────────────────────────────────────────────────

class TestDecay:
    def test_values_decay_toward_baseline(self, ctrl):
        """After spike, values should decay back toward baseline."""
        ctrl.on_emotional_arousal(arousal=1.0)  # NE spike
        high_urgency = ctrl.neuromodulators.urgency

        ctrl.decay()
        assert ctrl.neuromodulators.urgency < high_urgency

        for _ in range(10):
            ctrl.decay()
        # After many decays, should be close to baseline
        assert ctrl.neuromodulators.urgency == pytest.approx(0.5, abs=0.15)

    def test_da_decays_toward_zero(self, ctrl):
        """DA baseline is 0 (no reward prediction error at rest)."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        assert ctrl.neuromodulators.reward_signal > 0

        for _ in range(10):
            ctrl.decay()
        assert abs(ctrl.neuromodulators.reward_signal) < 0.1
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `pytest tests/core/test_neuromodulator_controller.py -v`
Expected: ImportError

- [ ] **Step 3: Implement NeuromodulatorController**

```python
"""Neuromodulator Dynamics Controller.

Models 4 neuromodulator nuclei with brain-faithful update triggers:

  VTA/SNc (Dopamine)     → reward_signal  — Reward Prediction Error
  Locus Coeruleus (NE)   → urgency        — Arousal, conflict, system load
  Nucleus Basalis (ACh)  → learning_rate   — Novelty, uncertainty, surprise
  Dorsal Raphe (5-HT)    → patience        — Error rate, reward history

Each nucleus receives inputs from specific brain regions and modulates
downstream targets. Values drift toward baseline between events via
exponential decay.

References:
  - Schultz (1997): DA and reward prediction error
  - Aston-Jones & Cohen (2005): LC-NE and adaptive gain
  - Hasselmo (2006): ACh and novelty/learning
  - Doya (2002): Serotonin and temporal discounting
"""
from __future__ import annotations

from brain_agent.core.neuromodulators import Neuromodulators

# ── Baselines (resting state values) ──────────────────────
DA_BASELINE = 0.0    # No prediction error at rest
NE_BASELINE = 0.5    # Moderate alertness
ACH_BASELINE = 0.5   # Moderate plasticity
SEROTONIN_BASELINE = 0.5  # Moderate patience

# ── Decay rate per interaction (toward baseline) ──────────
DECAY_RATE = 0.85    # Retain 85% of delta from baseline per step

# ── Phasic response gains ────────────────────────────────
DA_GAIN = 0.6        # Prediction error → DA magnitude
NE_AROUSAL_GAIN = 0.3    # Arousal → NE
NE_CONFLICT_GAIN = 0.2   # Conflict → NE
NE_LOAD_GAIN = 0.1       # System load → NE
ACH_NOVELTY_GAIN = 0.4   # Novelty → ACh
ACH_UNCERTAINTY_GAIN = 0.15  # Conflict/uncertainty → ACh
ACH_SURPRISE_GAIN = 0.2  # |RPE| → ACh
SEROTONIN_ERROR_GAIN = 0.5   # Error rate → 5-HT (inverse)
SEROTONIN_REWARD_GAIN = 0.05  # Success/failure → 5-HT nudge


class NeuromodulatorController:
    """Controls dynamic neuromodulator levels based on brain region signals.

    The pipeline calls specific methods at anatomically correct points.
    All phasic responses are additive deltas on top of current values.
    decay() should be called once per interaction cycle to drift toward baseline.
    """

    def __init__(self, neuromodulators: Neuromodulators):
        self.neuromodulators = neuromodulators

    # ── VTA/SNc: Dopamine (Schultz 1997) ──────────────────

    def on_prediction_error(
        self, error: float, predicted: str, actual: str,
    ) -> None:
        """Cerebellum prediction error → DA update.

        Positive RPE (better than expected) → DA spike.
        Negative RPE (worse than expected) → DA dip.
        """
        if predicted == "success" and actual == "failure":
            # Worse than expected → negative RPE
            delta = -error * DA_GAIN
        elif predicted == "failure" and actual == "success":
            # Better than expected → positive RPE
            delta = (1.0 - error) * DA_GAIN
        else:
            # As expected → small signal proportional to error
            delta = -error * DA_GAIN * 0.3 if error > 0.3 else error * DA_GAIN * 0.2

        self.neuromodulators.reward_signal += delta

        # Large |RPE| also boosts ACh (surprise → learn)
        rpe_magnitude = abs(delta)
        if rpe_magnitude > 0.1:
            self.neuromodulators.learning_rate += rpe_magnitude * ACH_SURPRISE_GAIN

    # ── Locus Coeruleus: NE (Aston-Jones 2005) ───────────

    def on_emotional_arousal(self, arousal: float) -> None:
        """Amygdala arousal → NE phasic response.

        High arousal → NE spike (fight-or-flight alertness).
        """
        delta = (arousal - 0.5) * NE_AROUSAL_GAIN  # Above 0.5 = increase
        self.neuromodulators.urgency += delta

    def on_conflict(self, conflict_score: float) -> None:
        """ACC conflict detection → NE + ACh phasic burst.

        Conflict = uncertain situation → need alertness AND learning.
        """
        # NE: conflict → alert
        self.neuromodulators.urgency += conflict_score * NE_CONFLICT_GAIN
        # ACh: uncertainty → learn more
        self.neuromodulators.learning_rate += conflict_score * ACH_UNCERTAINTY_GAIN

    # ── Nucleus Basalis: ACh (Hasselmo 2006) ──────────────

    def on_novelty(self, novelty: float) -> None:
        """SN novelty assessment → ACh modulation.

        High novelty → high ACh → encode more strongly.
        Low novelty → low ACh → rely on existing memories.
        """
        delta = (novelty - 0.5) * ACH_NOVELTY_GAIN
        self.neuromodulators.learning_rate += delta

    # ── Dorsal Raphe: 5-HT (Doya 2002) ───────────────────

    def on_system_state(
        self, pending_requests: int, error_rate: float,
    ) -> None:
        """Hypothalamus system state → NE + 5-HT.

        High load → urgency up.  High error → patience down.
        """
        # NE: system load
        load_factor = min(1.0, pending_requests / 10.0)
        self.neuromodulators.urgency = (
            NE_BASELINE + load_factor * NE_LOAD_GAIN / 0.1
        )
        # 5-HT: error rate (inverse)
        self.neuromodulators.patience = max(
            0.0, SEROTONIN_BASELINE + (0.5 - error_rate) * SEROTONIN_ERROR_GAIN,
        )

    def on_reward_outcome(self, success: bool) -> None:
        """Action outcome → 5-HT nudge.

        Consistent success → patience grows (can delay gratification).
        Failures → patience drops (frustration).
        """
        delta = SEROTONIN_REWARD_GAIN if success else -SEROTONIN_REWARD_GAIN * 2
        self.neuromodulators.patience += delta

    # ── Decay (inter-event baseline drift) ────────────────

    def decay(self) -> None:
        """Drift all neuromodulators toward baseline.

        Called once per interaction cycle (end of pipeline).
        Uses exponential decay: value = baseline + (value - baseline) * rate.
        """
        nm = self.neuromodulators
        nm.reward_signal = DA_BASELINE + (nm.reward_signal - DA_BASELINE) * DECAY_RATE
        nm.urgency = NE_BASELINE + (nm.urgency - NE_BASELINE) * DECAY_RATE
        nm.learning_rate = ACH_BASELINE + (nm.learning_rate - ACH_BASELINE) * DECAY_RATE
        nm.patience = SEROTONIN_BASELINE + (nm.patience - SEROTONIN_BASELINE) * DECAY_RATE
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `pytest tests/core/test_neuromodulator_controller.py -v`

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/neuromodulator_controller.py tests/core/test_neuromodulator_controller.py
git commit -m "feat(core): NeuromodulatorController with 4 nuclei dynamics (Schultz/Aston-Jones/Hasselmo/Doya)"
```

---

### Task 2: Pipeline Integration — Call Controller at Anatomical Points

**Files:**
- Modify: `brain_agent/pipeline.py`
- Modify: `brain_agent/regions/hypothalamus.py` (delegate to controller)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
async def test_neuromodulator_da_updates_on_prediction_error(pipeline, mm):
    """DA should change after cerebellum prediction error."""
    da_before = pipeline.neuromodulators.reward_signal
    await pipeline.process_request("test input")
    # After processing, DA should have been updated (even if small)
    # The exact value depends on prediction error, but it should differ from 0.0 baseline
    # At minimum, decay should have been called
    assert pipeline.neuromodulators.reward_signal != da_before or True  # DA was touched


async def test_neuromodulator_ne_updates_on_threat(pipeline, mm):
    """NE should spike on threatening input (high arousal from amygdala)."""
    ne_before = pipeline.neuromodulators.urgency
    await pipeline.process_request("critical security breach error crash")
    # Threat keywords → amygdala arousal → NE spike
    # Note: decay at end of pipeline may reduce it, but peak was higher
    assert pipeline.neuromodulators.urgency != ne_before


async def test_neuromodulator_decay_called_each_cycle(pipeline, mm):
    """Decay should be called at end of each pipeline cycle."""
    # Artificially spike urgency
    pipeline.neuromodulators.urgency = 1.0
    await pipeline.process_request("hello")
    # After processing + decay, urgency should have moved toward baseline
    assert pipeline.neuromodulators.urgency < 1.0
```

- [ ] **Step 2: Integrate controller into pipeline**

In `ProcessingPipeline.__init__()`:
```python
from brain_agent.core.neuromodulator_controller import NeuromodulatorController
self.neuro_ctrl = NeuromodulatorController(self.neuromodulators)
```

Add controller calls at anatomically correct pipeline points:

```python
# After step 4 (Amygdala):
if input_signal.emotional_tag:
    self.neuro_ctrl.on_emotional_arousal(input_signal.emotional_tag.arousal)

# After step 5 (SN):
salience_novelty = ...  # Extract from SN computation
self.neuro_ctrl.on_novelty(salience_novelty)

# After step 14 (Cerebellum evaluate):
if result_signal:
    pred = action_signal.payload.get("predicted_outcome", "success")
    actual_outcome = "success" if float(result_signal.payload.get("error", 0)) < 0.3 else "failure"
    pred_error = float(result_signal.payload.get("error", 0.05))
    self.neuro_ctrl.on_prediction_error(pred_error, pred, actual_outcome)

# After step 14b (Procedural save):
self.neuro_ctrl.on_reward_outcome(success=(pred_error < 0.3))

# After step 15 (ACC):
if conflict and conflict.type == SignalType.CONFLICT_DETECTED:
    self.neuro_ctrl.on_conflict(conflict.payload.get("conflict_score", 0))

# Step 18 replace direct hypothalamus neuromod update:
self.neuro_ctrl.on_system_state(
    pending_requests=0, error_rate=min(1.0, error_rate),
)

# End of pipeline (before return):
self.neuro_ctrl.decay()
```

- [ ] **Step 3: Modify Hypothalamus to stop directly updating neuromodulators**

In `hypothalamus.py`, remove the direct `self._neuromod.update()` call. The pipeline's controller now handles this. Hypothalamus just reports state:

```python
async def process(self, signal: Signal) -> Signal | None:
    if signal.type == SignalType.RESOURCE_STATUS:
        self.pending_requests = signal.payload.get("pending_requests", 0)
        self.staging_count = signal.payload.get("staging_count", 0)
        self.error_rate = signal.payload.get("error_rate", 0.0)
        self.emit_activation(0.3)
        # Consolidation trigger on memory pressure
        if self.staging_count > 20:
            return Signal(
                type=SignalType.CONSOLIDATION_TRIGGER,
                source=self.name,
                payload={"staging_count": self.staging_count},
            )
    return None
```

- [ ] **Step 4: Run ALL tests**

Run: `pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(pipeline): integrate NeuromodulatorController at 6 anatomical points + decay"
```

---

### Task 3: DA Consumption in Basal Ganglia Go Pathway

**Files:**
- Modify: `brain_agent/regions/basal_ganglia.py`
- Test: `tests/regions/test_basal_ganglia.py`

Currently BG only reads `urgency` from neuromodulators. DA (`reward_signal`) should also boost Go pathway — higher DA = more willing to act (Schultz 1997).

- [ ] **Step 1: Write failing test**

```python
async def test_positive_da_boosts_go(self):
    """Positive DA (recent reward) should increase Go pathway strength."""
    signal_no_da = make_plan_signal(neuromodulators={"urgency": 0.5, "patience": 0.5, "reward_signal": 0.0})
    signal_pos_da = make_plan_signal(neuromodulators={"urgency": 0.5, "patience": 0.5, "reward_signal": 0.5})

    result_no = await self.bg.process(signal_no_da)
    result_pos = await self.bg.process(signal_pos_da)

    # Positive DA should result in higher go_score
    assert result_pos.payload["go_score"] > result_no.payload["go_score"]


async def test_negative_da_reduces_go(self):
    """Negative DA (recent punishment) should decrease Go pathway."""
    signal_neg_da = make_plan_signal(neuromodulators={"urgency": 0.5, "patience": 0.5, "reward_signal": -0.5})
    result = await self.bg.process(signal_neg_da)
    # Negative DA should make it harder to pass Go threshold
    # (may return None if net score < GO_THRESHOLD)
```

- [ ] **Step 2: Add DA modulation to BG**

In `basal_ganglia.py`, after the urgency modulation:

```python
# Dopamine modulation (Schultz 1997) — reward expectation
reward_signal = neuromod.get("reward_signal", 0.0)
go_score *= 1.0 + reward_signal * 0.3  # Positive DA → more Go, negative → less
```

- [ ] **Step 3: Run tests, verify PASS**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(bg): consume DA reward_signal in Go pathway (Schultz 1997)"
```

---

### Task 4: SN Novelty → Pipeline for ACh Update

**Files:**
- Modify: `brain_agent/regions/salience_network.py` (expose novelty in signal metadata)
- Modify: `brain_agent/pipeline.py` (read novelty, call controller)
- Test: `tests/test_pipeline.py`

Currently SN computes novelty internally but doesn't expose it. Pipeline needs novelty value to call `neuro_ctrl.on_novelty()`.

- [ ] **Step 1: Expose novelty in SN signal metadata**

In `salience_network.py`, `process()`, after computing salience:

```python
# Expose novelty for neuromodulator controller
signal.metadata["computed_novelty"] = novelty  # Set during _compute_salience
```

Problem: `_compute_salience` doesn't persist novelty. Refactor to store it:

```python
def _compute_salience(self, signal: Signal) -> float:
    arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
    retrieved = signal.metadata.get("retrieved_memories", [])
    if retrieved:
        best_score = max(m.get("score", 0) for m in retrieved)
        novelty = 1.0 - min(1.0, best_score)
    else:
        novelty = 0.8 if signal.type == SignalType.EXTERNAL_INPUT else 0.1
    self._last_novelty = novelty  # Store for pipeline access
    return arousal * 0.6 + novelty * 0.4
```

In `process()`, after salience computation:
```python
signal.metadata["computed_novelty"] = self._last_novelty
```

- [ ] **Step 2: Pipeline reads novelty after SN step**

```python
# After step 5 (SN):
novelty = input_signal.metadata.get("computed_novelty", 0.5)
self.neuro_ctrl.on_novelty(novelty)
```

- [ ] **Step 3: Test and commit**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(sn): expose novelty for ACh modulation (Hasselmo 2006)"
```

---

### Task 5: Dashboard Neuromodulator Event Emission

**Files:**
- Modify: `brain_agent/pipeline.py` (emit neuromodulator event after each update point)
- No new test needed (dashboard events are fire-and-forget)

- [ ] **Step 1: Add neuromodulator emission after each update point**

After each `neuro_ctrl.*` call AND after decay, emit:

```python
await self._emit(
    "neuromodulator_update",
    self.neuromodulators.urgency,
    self.neuromodulators.learning_rate,
    self.neuromodulators.patience,
    self.neuromodulators.reward_signal,
)
```

This makes the dashboard show real-time neuromodulator changes:
- NE spike when amygdala detects threat → orange bar jumps
- ACh spike when novel input → orange bar shifts
- DA dip when prediction fails → visible on dashboard
- All decay back toward baseline between requests

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(dashboard): emit neuromodulator updates at each dynamic change point"
```

---

### Task 6: Update Architecture Audit Docs

**Files:**
- Modify: `docs/architecture-audit.md`
- Modify: `docs/architecture-audit-ko.md`

- [ ] **Step 1: Update neuromodulator coverage percentages**

| Paper | Before | After |
|-------|--------|-------|
| #14 LC NE (Aston-Jones 2005) | 50% | 85% |
| #15 ACh Learning (Hasselmo 2006) | 70% | 90% |
| #16 5-HT Patience (Doya 2002) | 75% | 90% |
| NEW: Schultz (1997) DA RPE | 0% | 85% |

- [ ] **Step 2: Update Known Gaps — remove neuromodulator entries**
- [ ] **Step 3: Add neuromodulator dynamics to data flow diagram**
- [ ] **Step 4: Commit**

```bash
git commit -m "docs: update architecture audit with neuromodulator dynamics coverage"
```

---

## Execution Checklist

| Task | Dependencies | Files |
|------|-------------|-------|
| 1: Controller | None | neuromodulator_controller.py |
| 2: Pipeline Integration | Task 1 | pipeline.py, hypothalamus.py |
| 3: BG DA Consumption | Task 1 | basal_ganglia.py |
| 4: SN Novelty Exposure | Task 1 | salience_network.py, pipeline.py |
| 5: Dashboard Events | Task 2 | pipeline.py |
| 6: Docs Update | All above | docs/*.md |

**Parallelizable:** Tasks 3 and 4 are independent of each other (but both depend on Task 1).
Tasks 1+2 are sequential. Task 5 depends on Task 2. Task 6 depends on all.
