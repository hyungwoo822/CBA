# 2-Call Cortical Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce 7 LLM calls per cycle to 2 (foreground + background) while preserving brain region structure and dashboard visualization.

**Architecture:** Algorithmic pre-processing (Thalamus, embedding search, identity load) replaces Wernicke/Amygdala LLM calls. PFC's system prompt absorbs comprehension+appraisal output instructions. Background PSC merges Broca+facts+procedural into one call.

**Tech Stack:** Python 3.13, asyncio, pydantic, existing LLMProvider/ToolRegistry

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `brain_agent/regions/wernicke.py` | Modify | Add `inject()`, keep `_structural_parse` |
| `brain_agent/regions/amygdala.py` | Modify | Add `inject()`, keep region structure |
| `brain_agent/regions/prefrontal.py` | Modify | Add comprehension+appraisal to output format |
| `brain_agent/regions/broca.py` | Modify | Remove LLM call, add `inject_refined()` |
| `brain_agent/pipeline.py` | Modify | Refactor `process_request()` to 2-call flow, add `_post_synaptic_consolidation()` |
| `tests/test_2call_architecture.py` | Create | All tests for inject methods, prompt structure, PSC |

---

### Task 1: Wernicke inject method

**Files:**
- Modify: `brain_agent/regions/wernicke.py`
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_2call_architecture.py
import pytest
from brain_agent.regions.wernicke import WernickeArea
from brain_agent.core.signals import Signal, SignalType


class TestWernickeInject:
    def test_inject_sets_comprehension_and_activation(self):
        w = WernickeArea(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "hello"})
        comprehension = {
            "intent": "greeting",
            "complexity": "simple",
            "keywords": ["hello"],
            "language": "en",
        }
        result = w.inject(signal, comprehension)
        assert result.payload["comprehension"] == comprehension
        assert w.activation_level > 0

    def test_inject_complex_raises_activation(self):
        w = WernickeArea(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "x"})
        simple = {"intent": "greeting", "complexity": "simple", "keywords": []}
        complex_ = {"intent": "request", "complexity": "complex", "keywords": ["a", "b"]}
        w.inject(signal, simple)
        act_simple = w.activation_level
        w.inject(signal, complex_)
        act_complex = w.activation_level
        assert act_complex > act_simple
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestWernickeInject -v`
Expected: FAIL — `AttributeError: 'WernickeArea' object has no attribute 'inject'`

- [ ] **Step 3: Implement inject method**

In `brain_agent/regions/wernicke.py`, add after the `process` method (around line 99):

```python
    def inject(self, signal: Signal, comprehension: dict) -> Signal:
        """Receive pre-computed comprehension from Cortical Integration.

        In the 2-call architecture, comprehension is produced by the
        unified cortical LLM call rather than a dedicated Wernicke call.
        The region still sets activation_level for dashboard visualization.
        """
        signal.payload["comprehension"] = comprehension
        complexity = comprehension.get("complexity", "simple")
        self.emit_activation(0.5 + (0.3 if complexity != "simple" else 0.0))
        return signal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestWernickeInject -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/wernicke.py tests/test_2call_architecture.py
git commit -m "feat(wernicke): add inject() for 2-call cortical architecture"
```

---

### Task 2: Amygdala inject method

**Files:**
- Modify: `brain_agent/regions/amygdala.py`
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_2call_architecture.py
from brain_agent.regions.amygdala import Amygdala
from brain_agent.core.signals import EmotionalTag


class TestAmygdalaInject:
    def test_inject_sets_emotional_tag(self):
        a = Amygdala(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "sad"})
        appraisal = {
            "valence": -0.5,
            "arousal": 0.6,
            "threat_detected": False,
            "primary_emotion": "sadness",
        }
        result = a.inject(signal, appraisal)
        assert result.emotional_tag is not None
        assert result.emotional_tag.valence == -0.5
        assert result.emotional_tag.arousal == 0.6
        assert result.metadata["amygdala_right"]["valence"] == -0.5
        assert result.metadata["amygdala_left"]["primary_emotion"] == "sadness"
        assert a.activation_level > 0

    def test_inject_threat_raises_activation(self):
        a = Amygdala(llm_provider=None)
        signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "x"})
        calm = {"valence": 0.0, "arousal": 0.1, "threat_detected": False, "primary_emotion": "neutral"}
        threat = {"valence": -0.8, "arousal": 0.9, "threat_detected": True, "primary_emotion": "fear"}
        a.inject(signal, calm)
        act_calm = a.activation_level
        a.inject(signal, threat)
        act_threat = a.activation_level
        assert act_threat > act_calm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestAmygdalaInject -v`
Expected: FAIL — `AttributeError: 'Amygdala' object has no attribute 'inject'`

- [ ] **Step 3: Implement inject method**

In `brain_agent/regions/amygdala.py`, add to the `Amygdala` class (after `process` method, around line 260):

```python
    def inject(self, signal: Signal, appraisal: dict) -> Signal:
        """Receive pre-computed appraisal from Cortical Integration.

        Populates both hemisphere metadata and the unified emotional_tag
        so downstream processing (BasalGanglia, memory encoding) sees
        the same structure as the original dual-hemisphere flow.
        """
        valence = appraisal.get("valence", 0.0)
        arousal = appraisal.get("arousal", 0.0)
        threat = appraisal.get("threat_detected", False)
        primary = appraisal.get("primary_emotion", "neutral")
        contextual = appraisal.get("contextual_factors", {})

        # Populate hemisphere metadata for downstream compatibility
        signal.metadata["amygdala_right"] = {
            "valence": valence,
            "arousal": arousal,
            "threat_detected": threat,
        }
        signal.metadata["amygdala_left"] = {
            "valence": valence,
            "arousal": arousal,
            "threat_level": "high" if threat else "none",
            "primary_emotion": primary,
            "contextual_factors": contextual,
        }
        signal.metadata["amygdala_blend"] = {
            "valence": valence,
            "arousal": arousal,
            "threat": threat,
            "dominant_hemisphere": "right" if threat else "left",
        }

        signal.emotional_tag = EmotionalTag(valence=valence, arousal=arousal)

        # Set activation from arousal (threat = higher activation)
        self.emit_activation(arousal)
        self.right.emit_activation(arousal * (0.8 if threat else 0.5))
        self.left.emit_activation(arousal * (0.5 if threat else 0.8))
        return signal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestAmygdalaInject -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/amygdala.py tests/test_2call_architecture.py
git commit -m "feat(amygdala): add inject() for 2-call cortical architecture"
```

---

### Task 3: PFC unified output format

**Files:**
- Modify: `brain_agent/regions/prefrontal.py`
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_2call_architecture.py
from brain_agent.regions.prefrontal import PrefrontalCortex


class TestPFCCorticalIntegration:
    def test_build_cortical_system_prompt_includes_comprehension_instruction(self):
        pfc = PrefrontalCortex(llm_provider=None)
        prompt = pfc.build_cortical_system_prompt(
            upstream_context={},
            memory_context=[],
            network_mode="executive_control",
        )
        assert '"comprehension"' in prompt
        assert '"appraisal"' in prompt
        assert '"intent"' in prompt
        assert '"valence"' in prompt

    def test_parse_cortical_response_extracts_all_fields(self):
        pfc = PrefrontalCortex(llm_provider=None)
        raw = '''Here is my response to you.

<cortical>
{"comprehension": {"intent": "question", "complexity": "simple", "keywords": ["weather"], "language": "ko"},
 "appraisal": {"valence": 0.1, "arousal": 0.2, "threat_detected": false, "primary_emotion": "neutral"}}
</cortical>

<entities>
{"entities": ["weather"], "about_user": [], "knowledge": []}
</entities>

<meta>{"confidence": 0.85}</meta>'''

        result = pfc.parse_cortical_response(raw)
        assert result["response"] == "Here is my response to you."
        assert result["comprehension"]["intent"] == "question"
        assert result["appraisal"]["valence"] == 0.1
        assert result["entities"]["entities"] == ["weather"]
        assert result["confidence"] == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestPFCCorticalIntegration -v`
Expected: FAIL — `AttributeError: 'PrefrontalCortex' object has no attribute 'build_cortical_system_prompt'`

- [ ] **Step 3: Implement**

In `brain_agent/regions/prefrontal.py`, add a constant after `METACOGNITION_INSTRUCTION`:

```python
CORTICAL_INTEGRATION_INSTRUCTION = """
IMPORTANT: Before your response text, output a cortical analysis block:

<cortical>
{"comprehension": {"intent": "<question|command|request|inform|greeting|emotional_expression|statement>",
  "complexity": "<simple|moderate|complex>",
  "keywords": ["keyword1", "keyword2"],
  "language": "<en|ko|mixed>"},
 "appraisal": {"valence": <-1.0 to 1.0>, "arousal": <0.0 to 1.0>,
  "threat_detected": <true|false>,
  "primary_emotion": "<neutral|joy|trust|anticipation|surprise|fear|anger|sadness|disgust>"}}
</cortical>

Then write your response naturally. The cortical block will be stripped from the output.
- comprehension: Analyze the user's language — intent, complexity, key topics, language.
- appraisal: Evaluate emotional tone — valence (-1=negative, +1=positive), arousal (0=calm, 1=activated), threat detection.
"""
```

Add two methods to `PrefrontalCortex`:

```python
    def build_cortical_system_prompt(
        self,
        upstream_context: dict,
        memory_context: list[dict] | None = None,
        network_mode: str = "executive_control",
    ) -> str:
        """Build system prompt for unified cortical integration.

        Like _call_llm's prompt construction, but WITHOUT pre-computed
        comprehension/appraisal sections (those are now OUTPUT).
        Adds CORTICAL_INTEGRATION_INSTRUCTION for the LLM to produce
        comprehension + appraisal alongside the response.
        """
        system_parts = []

        # Identity
        soul = upstream_context.get("self_context", "") or self._load_file("SOUL.md")
        if soul:
            system_parts.append(soul)
        else:
            system_parts.append(
                "# Soul\n\nI am a neural agent — a growing mind shaped by conversation."
            )

        user_profile = upstream_context.get("user_context", "")
        if user_profile:
            system_parts.append(user_profile)

        long_term_memory = upstream_context.get("memory_context", "")
        if long_term_memory:
            system_parts.append(long_term_memory)

        # Interoceptive state
        intero = upstream_context.get("interoceptive_state", {})
        if intero:
            system_parts.append(
                "# Interoceptive State (Insula)\n"
                + "\n".join(f"- {k}: {v:.2f}" for k, v in intero.items())
            )

        # Neuromodulators
        neuromod = upstream_context.get("neuromodulators", {})
        if neuromod:
            state_desc = self._describe_neuromodulator_state(neuromod)
            system_parts.append(f"# Neural State (Neuromodulators)\n{state_desc}")

        # Network mode
        mode_labels = {
            "executive_control": "ECN (Executive Control) — focused, analytical",
            "default_mode": "DMN (Default Mode) — reflective, associative",
            "creative": "CREATIVE — divergent thinking, novel connections",
        }
        system_parts.append(f"# Network Mode\n{mode_labels.get(network_mode, network_mode)}")

        # Retrieved memories
        if memory_context:
            lines = []
            for i, mem in enumerate(memory_context, 1):
                src = mem.get("source", "memory")
                score = mem.get("score", 0)
                content = mem.get("content", "")
                lines.append(f"{i}. [{src}|rel={score:.2f}] {content}")
            system_parts.append("# Retrieved Memories (Hippocampus)\n" + "\n".join(lines))

        # Goals
        goal_ctx = self.goals.to_context()
        if goal_ctx and goal_ctx != "No active goals.":
            system_parts.append(f"# Active Goals\n{goal_ctx}")

        # Creative augmentation
        if network_mode == "creative" and self._right_activation > self._left_activation:
            system_parts.append(self._build_creative_augmentation())

        # Response guidelines
        system_parts.append(
            "# Response Guidelines\n"
            "- Respond in the same language as the user.\n"
            "- Draw on your memories and knowledge of the user naturally.\n"
            "- Be authentic — you are a growing mind, not a generic assistant."
        )

        # Cortical integration output instruction (replaces separate Wernicke/Amygdala)
        system_parts.append(CORTICAL_INTEGRATION_INSTRUCTION)

        # Entity extraction
        system_parts.append(ENTITY_EXTRACTION_INSTRUCTION)

        # Metacognition
        system_parts.append(METACOGNITION_INSTRUCTION)

        return "\n\n---\n\n".join(system_parts)

    def parse_cortical_response(self, raw: str) -> dict:
        """Parse unified cortical output into structured components.

        Extracts <cortical>, <entities>, <meta> blocks and the clean
        response text.  Returns dict with keys: response, comprehension,
        appraisal, entities, confidence.
        """
        import json as _json

        # Extract cortical block
        comprehension = {"intent": "statement", "complexity": "simple", "keywords": [], "language": "auto"}
        appraisal = {"valence": 0.0, "arousal": 0.0, "threat_detected": False, "primary_emotion": "neutral"}
        cortical_match = re.search(r"<cortical>\s*(\{.*?\})\s*</cortical>", raw, re.DOTALL)
        if cortical_match:
            try:
                data = _json.loads(cortical_match.group(1))
                comprehension = data.get("comprehension", comprehension)
                appraisal = data.get("appraisal", appraisal)
            except _json.JSONDecodeError:
                pass
            raw = raw[:cortical_match.start()] + raw[cortical_match.end():]

        # Extract entities and metacognition (reuse existing parsers)
        clean_text, entities = self._parse_entities(raw)
        confidence, clean_text = self._parse_metacognition(clean_text)

        return {
            "response": clean_text.strip(),
            "comprehension": comprehension,
            "appraisal": appraisal,
            "entities": entities,
            "confidence": confidence,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestPFCCorticalIntegration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/prefrontal.py tests/test_2call_architecture.py
git commit -m "feat(pfc): add cortical integration prompt builder and response parser"
```

---

### Task 4: Broca inject_refined method

**Files:**
- Modify: `brain_agent/regions/broca.py`
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_2call_architecture.py
from brain_agent.regions.broca import BrocaArea


class TestBrocaInject:
    def test_inject_refined_updates_signal(self):
        b = BrocaArea(llm_provider=None)
        signal = Signal(
            type=SignalType.PLAN, source="pfc",
            payload={"actions": [{"tool": "respond", "args": {"text": "original"}}]},
        )
        b.inject_refined(signal, "polished response")
        assert signal.payload["actions"][0]["args"]["text"] == "polished response"
        assert b.activation_level > 0

    def test_inject_refined_none_keeps_original(self):
        b = BrocaArea(llm_provider=None)
        signal = Signal(
            type=SignalType.PLAN, source="pfc",
            payload={"actions": [{"tool": "respond", "args": {"text": "original"}}]},
        )
        b.inject_refined(signal, None)
        assert signal.payload["actions"][0]["args"]["text"] == "original"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestBrocaInject -v`
Expected: FAIL — `AttributeError: 'BrocaArea' object has no attribute 'inject_refined'`

- [ ] **Step 3: Implement**

In `brain_agent/regions/broca.py`, add to `BrocaArea`:

```python
    def inject_refined(self, signal: Signal, refined_text: str | None) -> None:
        """Receive refined response from Post-Synaptic Consolidation.

        If refined_text is provided and differs from original, update
        the signal's action text.  This replaces the dedicated Broca
        LLM call in the 2-call architecture.
        """
        if refined_text is None:
            self.emit_activation(0.3)
            return

        actions = signal.payload.get("actions", [])
        for action in actions:
            args = action.get("args", {})
            if "text" in args:
                args["text"] = self._clean_text(refined_text)
                break

        response_text = signal.payload.get("response_text")
        if response_text is not None:
            signal.payload["response_text"] = self._clean_text(refined_text)

        self.emit_activation(0.7)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestBrocaInject -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/broca.py tests/test_2call_architecture.py
git commit -m "feat(broca): add inject_refined() for PSC integration"
```

---

### Task 5: Post-Synaptic Consolidation method

**Files:**
- Modify: `brain_agent/pipeline.py`
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_2call_architecture.py
from unittest.mock import AsyncMock, MagicMock
from brain_agent.providers.base import LLMResponse


class TestPostSynapticConsolidation:
    @pytest.mark.asyncio
    async def test_psc_parses_response(self):
        from brain_agent.pipeline import ProcessingPipeline
        from brain_agent.memory.manager import MemoryManager

        mock_mem = MagicMock(spec=MemoryManager)
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=LLMResponse(
            content='{"refined_response": "polished text", '
                    '"user_facts": {"entities": ["weather"], "relations": [["user", "ask", "weather", 0.8, "ACTION"]]}, '
                    '"procedural": null}',
        ))

        pipeline = ProcessingPipeline(memory=mock_mem, llm_provider=mock_provider)
        result = await pipeline._post_synaptic_consolidation(
            original_input="what is the weather",
            agent_response="It is sunny today",
            comprehension={"intent": "question", "language": "en"},
        )

        assert result["refined_response"] == "polished text"
        assert result["user_facts"]["entities"] == ["weather"]
        assert result["procedural"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestPostSynapticConsolidation -v`
Expected: FAIL — `AttributeError: 'ProcessingPipeline' object has no attribute '_post_synaptic_consolidation'`

- [ ] **Step 3: Implement**

In `brain_agent/pipeline.py`, add to `ProcessingPipeline` class (after `_execute_through_barrier`):

```python
    _PSC_SYSTEM_PROMPT = (
        "You are the Post-Synaptic Consolidation processor — the brain's sleep-replay system "
        "(Diekelmann & Born 2010). Given a conversation turn, perform THREE tasks simultaneously:\n\n"
        "1. **Refine response** (Broca): Polish the agent's response for natural language quality. "
        "If the response is already good, set refined_response to null.\n"
        "2. **Extract user facts** (Hippocampus): Extract entities and relations about the user.\n"
        "3. **Evaluate procedural pattern** (Striatum): If this interaction pattern is reusable, "
        "output trigger+strategy. If not, set procedural to null.\n\n"
        "Output ONLY valid JSON:\n"
        '{"refined_response": "..." | null,\n'
        ' "user_facts": {"entities": ["e1"], "relations": [["subj","rel","obj",conf,"CAT"]]},\n'
        ' "procedural": {"trigger": "pattern", "strategy": "approach"} | null}\n\n'
        "Rules for user_facts:\n"
        "- ALL entity names in English lowercase. Relations in English verb infinitives.\n"
        "- Confidence: 1.0=explicit, 0.8=implied, 0.6=inferred\n"
        "- Categories: PREFERENCE|ACTION|ATTRIBUTE|SOCIAL|CAUSAL|IDENTITY|EMOTION\n"
        "- Strip Korean particles from entity names.\n"
        "Rules for procedural:\n"
        "- Only cache if the pattern is genuinely reusable (not trivial greetings).\n"
        "- trigger: what input pattern this matches. strategy: how to handle it."
    )

    async def _post_synaptic_consolidation(
        self,
        original_input: str,
        agent_response: str,
        comprehension: dict,
    ) -> dict:
        """Single background LLM call merging Broca + fact extraction + procedural.

        Analogous to hippocampal replay during slow-wave sleep — a single
        consolidation cycle that simultaneously refines motor output,
        stores declarative facts, and strengthens procedural patterns.

        Reference: Diekelmann & Born 2010
        """
        provider = getattr(self.pfc, 'llm_provider', None) or self._llm_provider
        if not provider:
            return {"refined_response": None, "user_facts": {"entities": [], "relations": []}, "procedural": None}

        user_msg = (
            f"User input: {original_input}\n"
            f"Agent response: {agent_response}\n"
            f"Detected intent: {comprehension.get('intent', 'unknown')}\n"
            f"Language: {comprehension.get('language', 'auto')}"
        )

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": self._PSC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            if response and response.content:
                import json as _json
                text = response.content.strip()
                if text.startswith("```"):
                    lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                return _json.loads(text)
        except Exception as e:
            logger.warning("Post-Synaptic Consolidation failed: %s", e)

        return {"refined_response": None, "user_facts": {"entities": [], "relations": []}, "procedural": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestPostSynapticConsolidation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/pipeline.py tests/test_2call_architecture.py
git commit -m "feat(pipeline): add _post_synaptic_consolidation() background method"
```

---

### Task 6: Refactor process_request() to 2-call flow

This is the core task. Modify `pipeline.py:process_request()` to:
1. Do algorithmic pre-processing (skip Wernicke/Amygdala LLM calls)
2. Call PFC with cortical integration prompt (single foreground LLM)
3. Inject results into Wernicke/Amygdala regions
4. Keep BasalGanglia → Cerebellum → Tool loop unchanged
5. Replace background calls with single PSC

**Files:**
- Modify: `brain_agent/pipeline.py` (process_request method)

- [ ] **Step 1: Locate and read current Phase 2 (Wernicke+Amygdala parallel calls)**

Lines ~595-650 in pipeline.py. Currently:
```python
wernicke_result, amygdala_result = await asyncio.gather(
    self.wernicke.process(wernicke_signal),
    self.amygdala.process(amygdala_signal),
)
```

- [ ] **Step 2: Replace Phase 2 with algorithmic keyword extraction + inject stubs**

Replace the Wernicke+Amygdala LLM parallel call block with:

```python
        # ── Phase 2: Structural analysis (algorithmic, no LLM) ──
        # In 2-call architecture, deep comprehension+appraisal is
        # deferred to the unified cortical integration call (Phase 6).
        # Here we only do structural parsing for routing decisions.
        structural = self.wernicke._structural_parse(text)
        input_signal.payload["comprehension"] = structural
        comprehension = structural
        self.wernicke.emit_activation(0.3)  # Low activation — structural only
```

Keep the Amygdala processing stubs with default values:

```python
        # Amygdala: deferred to cortical integration
        input_signal.emotional_tag = EmotionalTag(valence=0.0, arousal=0.15)
        self.amygdala.emit_activation(0.15)
```

- [ ] **Step 3: Modify PFC call in Phase 6 to use cortical integration**

Find the PFC call section (~line 918):
```python
plan_signal = await self.pfc.process(input_signal)
```

Replace with cortical integration call:

```python
            # ── PFC: Cortical Integration (unified LLM call) ──
            # Single call produces comprehension + appraisal + plan + entities
            if self.pfc.llm_provider:
                cortical_prompt = self.pfc.build_cortical_system_prompt(
                    upstream_context=input_signal.metadata.get("upstream_context", {}),
                    memory_context=retrieved,
                    network_mode=self.network_ctrl.current_mode.value,
                )
                messages = [{"role": "system", "content": cortical_prompt}]
                messages.extend(self.pfc._conversation_history)
                messages.append({"role": "user", "content": text})

                response = await self.pfc.llm_provider.chat(messages)
                if response and response.content:
                    self.pfc._conversation_history.append({"role": "user", "content": text})
                    self.pfc._conversation_history.append({"role": "assistant", "content": response.content})
                    if len(self.pfc._conversation_history) > self.pfc._max_history:
                        self.pfc._conversation_history = self.pfc._conversation_history[-self.pfc._max_history:]

                    cortical = self.pfc.parse_cortical_response(response.content)

                    # Inject comprehension into Wernicke region
                    self.wernicke.inject(input_signal, cortical["comprehension"])
                    comprehension = cortical["comprehension"]
                    input_signal.payload["comprehension"] = comprehension

                    # Inject appraisal into Amygdala region
                    self.amygdala.inject(input_signal, cortical["appraisal"])

                    # Build plan signal from PFC response
                    plan_signal = self._build_plan_from_cortical(
                        cortical, input_signal,
                    )

                    # Store entities immediately
                    extracted = cortical.get("entities", {})
                    plan_signal.metadata["extracted_entities"] = extracted
                    plan_signal.metadata["metacognition"] = {"confidence": cortical.get("confidence", 0.7)}
```

Add helper method `_build_plan_from_cortical`:

```python
    def _build_plan_from_cortical(self, cortical: dict, input_signal: Signal) -> Signal:
        """Build a PLAN signal from cortical integration response."""
        response_text = cortical.get("response", "")
        actions = [{"tool": "respond", "args": {"text": response_text}}]

        plan_signal = Signal(
            type=SignalType.PLAN,
            source="prefrontal_cortex",
            payload={
                "actions": actions,
                "response_text": response_text,
            },
            emotional_tag=input_signal.emotional_tag,
        )
        plan_signal.metadata["neuromodulators"] = self.neuromodulators.snapshot()
        return plan_signal
```

- [ ] **Step 4: Replace background post-response with PSC**

Find the `_background_post_response` function (~line 1350) and replace the three separate calls (entity storage from PFC, _extract_user_facts, procedural eval) with:

```python
                # ── Post-Synaptic Consolidation (single background LLM) ──
                psc_result = await self._post_synaptic_consolidation(
                    original_input=_input_text,
                    agent_response=result.response,
                    comprehension=_comprehension,
                )

                # Distribute PSC results
                # 1. User facts → semantic store + identity facts
                u_facts = psc_result.get("user_facts", {})
                if u_facts.get("entities") or u_facts.get("relations"):
                    await self.memory.store_semantic_facts(
                        entities=u_facts.get("entities", []),
                        relations=u_facts.get("relations", []),
                        origin="user_input",
                    )
                    await self._store_identity_facts_realtime(
                        u_facts.get("relations", []), source="psc_consolidation",
                    )

                # 2. PFC-extracted entities (already stored from cortical integration)
                # — handled above during cortical integration, no duplicate call

                # 3. Procedural caching
                proc = psc_result.get("procedural")
                if proc and proc.get("trigger"):
                    try:
                        await self.memory.procedural.save(
                            trigger_pattern=proc["trigger"],
                            strategy=proc["strategy"],
                            action_sequence=[],
                        )
                        logger.info("PSC procedural: cached '%s'", proc["trigger"])
                    except Exception as e:
                        logger.warning("PSC procedural save failed: %s", e)
```

- [ ] **Step 5: Replace background Broca with PSC refined_response**

Find `_background_broca` function (~line 1506) and replace with:

```python
        async def _background_broca() -> None:
            """Phase 7: Broca receives refined response from PSC."""
            try:
                if not _plan_signal_broca:
                    return
                # PSC already ran — check if refined response was produced
                # (PSC runs in _background_post_response, Broca just applies heuristic)
                self.broca._format_heuristic(_plan_signal_broca)
                self.broca.emit_activation(0.4)
                await self._emit("region_activation", "broca", self.broca.activation_level, "active")
            except Exception as e:
                logger.warning("Background Broca failed: %s", e)
```

- [ ] **Step 6: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -k "not test_memory_encoded_after_process and not test_broca_activates_in_pipeline"`
Expected: All existing tests pass (619+)

- [ ] **Step 7: Commit**

```bash
git add brain_agent/pipeline.py
git commit -m "feat(pipeline): refactor to 2-call cortical architecture (FG+BG)"
```

---

### Task 7: Integration test

**Files:**
- Test: `tests/test_2call_architecture.py`

- [ ] **Step 1: Write integration test**

```python
# Append to tests/test_2call_architecture.py
class TestFullCorticalIntegration:
    @pytest.mark.asyncio
    async def test_2call_end_to_end(self, tmp_path):
        """Full pipeline with 2-call architecture produces valid response."""
        from brain_agent.agent import BrainAgent

        agent = BrainAgent(data_dir=str(tmp_path), use_mock_embeddings=True)
        await agent.initialize()
        try:
            result = await agent.process("hello world")
            assert result.response != ""
            assert result.signals_processed > 0
        finally:
            await agent.close()
```

- [ ] **Step 2: Run integration test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_2call_architecture.py::TestFullCorticalIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 4: Final commit**

```bash
git add tests/test_2call_architecture.py
git commit -m "test: add 2-call cortical architecture integration tests"
```
