# Question / Expression Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two interaction modes (Question / Expression) to the brain agent with a dashboard HUD toggle.

**Architecture:** Mode is passed as a parameter from the dashboard UI through the API to the pipeline. In Question mode, PFC receives gap-analysis instructions and memory timestamps. In Expression mode, all memory write paths are skipped and PFC is restricted to fact-only responses.

**Tech Stack:** React + Zustand (dashboard), FastAPI (server), Python pipeline + PFC prompt

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/src/stores/brainState.ts` | Add `interactionMode` state + setter, include in `submitChat` |
| `dashboard/src/components/InteractionModeToggle.tsx` | New: pill toggle above lightbulb |
| `dashboard/src/App.tsx` | Mount InteractionModeToggle |
| `brain_agent/dashboard/server.py` | Extract `mode` from form data |
| `brain_agent/agent.py` | Forward `interaction_mode` to pipeline |
| `brain_agent/pipeline.py` | Branch on `interaction_mode`: skip writes in expression |
| `brain_agent/regions/prefrontal.py` | Mode-specific prompt blocks + timestamp in memories |

---

### Task 1: Zustand Store — interactionMode

**Files:**
- Modify: `dashboard/src/stores/brainState.ts`

- [ ] **Step 1: Add state and action to BrainState interface**

In the `BrainState` interface (after `chatLoading: boolean`), add:

```typescript
  interactionMode: 'question' | 'expression'
  setInteractionMode: (mode: 'question' | 'expression') => void
```

- [ ] **Step 2: Add defaults and implementation**

In the `create<BrainState>` initializer (after `chatLoading: false`), add:

```typescript
  interactionMode: 'question',
  setInteractionMode: (mode) => set({ interactionMode: mode }),
```

- [ ] **Step 3: Include mode in submitChat**

In the `submitChat` function, after `fd.append('text', text)` (around line 252), add:

```typescript
      fd.append('mode', useBrainStore.getState().interactionMode)
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/stores/brainState.ts
git commit -m "feat(dashboard): add interactionMode to Zustand store"
```

---

### Task 2: InteractionModeToggle Component

**Files:**
- Create: `dashboard/src/components/InteractionModeToggle.tsx`

- [ ] **Step 1: Create the component**

```tsx
import { useBrainStore } from '../stores/brainState'

export function InteractionModeToggle() {
  const mode = useBrainStore((s) => s.interactionMode)
  const setMode = useBrainStore((s) => s.setInteractionMode)

  return (
    <div style={{
      position: 'fixed',
      top: '12%',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 901,
      display: 'flex',
      borderRadius: 20,
      overflow: 'hidden',
      border: '1px solid rgba(148,163,184,0.15)',
      background: 'rgba(10,10,20,0.7)',
      backdropFilter: 'blur(12px)',
      fontSize: 12,
      fontFamily: 'inherit',
      userSelect: 'none',
    }}>
      <button
        onClick={() => setMode('question')}
        style={{
          padding: '5px 14px',
          border: 'none',
          cursor: 'pointer',
          background: mode === 'question' ? 'rgba(249,115,22,0.25)' : 'transparent',
          color: mode === 'question' ? '#f97316' : 'rgba(226,232,240,0.5)',
          fontWeight: mode === 'question' ? 600 : 400,
          transition: 'all 0.2s ease',
          boxShadow: mode === 'question' ? '0 0 12px rgba(249,115,22,0.2)' : 'none',
        }}
      >
        Question
      </button>
      <button
        onClick={() => setMode('expression')}
        style={{
          padding: '5px 14px',
          border: 'none',
          cursor: 'pointer',
          background: mode === 'expression' ? 'rgba(96,165,250,0.25)' : 'transparent',
          color: mode === 'expression' ? '#60a5fa' : 'rgba(226,232,240,0.5)',
          fontWeight: mode === 'expression' ? 600 : 400,
          transition: 'all 0.2s ease',
          boxShadow: mode === 'expression' ? '0 0 12px rgba(96,165,250,0.2)' : 'none',
        }}
      >
        Expression
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Mount in App.tsx**

In `dashboard/src/App.tsx`, add import:

```typescript
import { InteractionModeToggle } from './components/InteractionModeToggle'
```

Inside the `App` component return, after `<KnowledgeGraphModal />` (line 254), add:

```tsx
      <InteractionModeToggle />
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/InteractionModeToggle.tsx dashboard/src/App.tsx
git commit -m "feat(dashboard): add Question/Expression toggle above lightbulb"
```

---

### Task 3: API Server — Accept mode Parameter

**Files:**
- Modify: `brain_agent/dashboard/server.py`

- [ ] **Step 1: Add mode parameter to endpoint**

In `server.py`, change the `process_message` function signature (line 321-323) from:

```python
    async def process_message(
        text: str = Form(default=""),
        files: list[UploadFile] = File(default=[]),
    ):
```

to:

```python
    async def process_message(
        text: str = Form(default=""),
        mode: str = Form(default="question"),
        files: list[UploadFile] = File(default=[]),
    ):
```

- [ ] **Step 2: Pass mode to agent.process()**

Change line 354 from:

```python
        result = await agent_inst.process(full_text, image=image_bytes, audio=audio_bytes)
```

to:

```python
        result = await agent_inst.process(full_text, image=image_bytes, audio=audio_bytes, interaction_mode=mode)
```

- [ ] **Step 3: Commit**

```bash
git add brain_agent/dashboard/server.py
git commit -m "feat(api): accept interaction_mode parameter in /api/process"
```

---

### Task 4: BrainAgent — Forward mode to Pipeline

**Files:**
- Modify: `brain_agent/agent.py`

- [ ] **Step 1: Add interaction_mode to process() signature**

Change the `process` method signature (line 156) from:

```python
    async def process(self, text: str, image: bytes | None = None, audio: bytes | None = None) -> PipelineResult:
```

to:

```python
    async def process(self, text: str, image: bytes | None = None, audio: bytes | None = None, interaction_mode: str = "question") -> PipelineResult:
```

- [ ] **Step 2: Pass to pipeline**

Change the `_pipeline_core` function (line 203-211) from:

```python
        async def _pipeline_core(ctx: MiddlewareContext) -> MiddlewareContext:
            result = await self.pipeline.process_request(
                ctx["user_input"],
                image=ctx.get("image"),
                audio=ctx.get("audio"),
                trace_run=trace_run,
            )
            ctx["result"] = result
            return ctx
```

to:

```python
        async def _pipeline_core(ctx: MiddlewareContext) -> MiddlewareContext:
            result = await self.pipeline.process_request(
                ctx["user_input"],
                image=ctx.get("image"),
                audio=ctx.get("audio"),
                trace_run=trace_run,
                interaction_mode=interaction_mode,
            )
            ctx["result"] = result
            return ctx
```

- [ ] **Step 3: Commit**

```bash
git add brain_agent/agent.py
git commit -m "feat(agent): forward interaction_mode to pipeline"
```

---

### Task 5: PFC — Mode-Specific Prompt Blocks

**Files:**
- Modify: `brain_agent/regions/prefrontal.py`

- [ ] **Step 1: Add mode prompt constants**

After `METACOGNITION_INSTRUCTION` (around line 71), add:

```python
QUESTION_MODE_INSTRUCTION = """
# Memory Gap Analysis (Question Mode)
You analyze retrieved memories for logical gaps and ask about them naturally during conversation.

Question targets:
1. Time-based contradictions — facts that conflict across dates.
   e.g., "Sick on 04-07" + "Went hiking on 04-10" → "저번에 아프다고 했는데 지금은 괜찮아진 거야?"
2. Causal/preference unknowns — events with unconfirmed emotional or preference consequences.
   e.g., "Hospitalized because of mussel" + no mussel preference recorded → "홍합 때문에 고생했는데, 이제 홍합 싫어졌어?"
3. Emotional/rational change — past memory vs current statement inconsistency.
   e.g., "Used to like coffee" + "Haven't mentioned coffee recently" → "예전에 커피 좋아한다고 했는데 요즘은 안 마셔?"

Rules:
- Maximum 1 question per turn. Only ask when a genuine gap exists.
- Do NOT ask tail-chasing questions (continuing a conversation the user already ended).
- Do NOT ask about missing details of past events (e.g., "What treatment did you get at the hospital?").
- Weave the question naturally into your response — not as a separate interrogation.
- If no gap is detected, respond normally without forcing a question.
"""

EXPRESSION_MODE_INSTRUCTION = """
# Expression Mode (Read-Only Memory)
You answer ONLY from facts stored in your memory. You are a faithful mirror of what you know.

Rules:
- If a fact is not in your memory, say "그건 아직 모르겠어" or equivalent.
- NEVER infer, extrapolate, or guess. No "아마", "~일 수도", "~겠지" expressions.
- NEVER fill logical gaps with reasoning. If memory says "user visited hospital" but not why, do NOT speculate about the reason.
- Only state what is explicitly recorded in Retrieved Memories and User Profile.
- Be honest about the boundaries of your knowledge.
"""
```

- [ ] **Step 2: Add interaction_mode parameter to build_cortical_system_prompt**

Change the method signature (line 226-230) from:

```python
    def build_cortical_system_prompt(
        self,
        upstream_context: dict,
        memory_context: list[dict] | None = None,
        network_mode: str = "executive_control",
    ) -> str:
```

to:

```python
    def build_cortical_system_prompt(
        self,
        upstream_context: dict,
        memory_context: list[dict] | None = None,
        network_mode: str = "executive_control",
        interaction_mode: str = "question",
    ) -> str:
```

- [ ] **Step 3: Append mode instruction before return**

In `build_cortical_system_prompt`, just before the `return` statement (line 326), replace:

```python
        return "\n\n---\n\n".join(system_parts)
```

with:

```python
        # Interaction mode instruction
        if interaction_mode == "expression":
            system_parts.append(EXPRESSION_MODE_INSTRUCTION)
        else:
            system_parts.append(QUESTION_MODE_INSTRUCTION)

        return "\n\n---\n\n".join(system_parts)
```

- [ ] **Step 4: Add timestamp to Retrieved Memories rendering**

In the same method, change the memory rendering block (around line 296) from:

```python
                lines.append(f"{i}. [{src}|rel={score:.2f}]{em_str} {content}")
```

to:

```python
                ts = mem.get("timestamp", "")
                ts_str = ""
                if ts:
                    # Extract date portion from ISO timestamp
                    ts_str = f"|{ts[:10]}" if len(ts) >= 10 else ""
                lines.append(f"{i}. [{src}|rel={score:.2f}{ts_str}]{em_str} {content}")
```

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/prefrontal.py
git commit -m "feat(pfc): add Question/Expression mode prompt blocks with memory timestamps"
```

---

### Task 6: Pipeline — Mode Branching

**Files:**
- Modify: `brain_agent/pipeline.py`

- [ ] **Step 1: Add interaction_mode to process_request signature**

Change line 612 from:

```python
    async def process_request(self, text: str = "", image: bytes | None = None, audio: bytes | None = None, trace_run=None) -> PipelineResult:
```

to:

```python
    async def process_request(self, text: str = "", image: bytes | None = None, audio: bytes | None = None, trace_run=None, interaction_mode: str = "question") -> PipelineResult:
```

- [ ] **Step 2: Pass interaction_mode to PFC prompt builder**

Find where `build_cortical_system_prompt` is called (around line 1096). Change from:

```python
                cortical_prompt = self.pfc.build_cortical_system_prompt(
                    upstream_context=input_signal.metadata.get("upstream_context", {}),
                    memory_context=retrieved,
                    network_mode=self.network_ctrl.current_mode.value,
                )
```

to:

```python
                cortical_prompt = self.pfc.build_cortical_system_prompt(
                    upstream_context=input_signal.metadata.get("upstream_context", {}),
                    memory_context=retrieved,
                    network_mode=self.network_ctrl.current_mode.value,
                    interaction_mode=interaction_mode,
                )
```

- [ ] **Step 3: Include timestamp in retrieve results**

In the `retrieve` method of `manager.py`, the episodic candidates (around line 390-399) already have `ep.get("timestamp")` available via the row data. Add `timestamp` to the candidate dict. In `brain_agent/memory/manager.py`, in the episodic candidate append block (around line 390), change from:

```python
            candidates.append({
                "id": ep["id"],
                "content": ep["content"],
                "source": "episodic",
                "relevance": relevance * retention,
                "importance": arousal,
                "access_count": ep.get("access_count", 0),
                "recency_distance": recency_dist,
                "context_similarity": ctx_sim,
            })
```

to:

```python
            candidates.append({
                "id": ep["id"],
                "content": ep["content"],
                "source": "episodic",
                "relevance": relevance * retention,
                "importance": arousal,
                "access_count": ep.get("access_count", 0),
                "recency_distance": recency_dist,
                "context_similarity": ctx_sim,
                "timestamp": ep.get("timestamp", ""),
            })
```

Also add `timestamp` for semantic candidates (around line 356-365). Semantic store results from ChromaDB don't have timestamps, so use empty string:

```python
            candidates.append({
                "id": mem["id"],
                "content": mem["content"],
                "source": "semantic",
                "relevance": relevance,
                "importance": mem.get("metadata", {}).get("strength", 0.5),
                "access_count": int(mem.get("metadata", {}).get("access_count", 0)),
                "recency_distance": 0.0,
                "context_similarity": relevance,
                "timestamp": "",
            })
```

- [ ] **Step 4: Skip background writes in Expression mode**

In the `_background_post_response` function definition area (around line 1544), capture `interaction_mode`:

After the existing `_trace_run = self._current_trace_run` line, add:

```python
        _interaction_mode = interaction_mode
```

Then inside `_background_post_response`, wrap the entire body after encoding in a mode check. Replace the current Phase 4 encoding block:

```python
            try:
                # ── Phase 4: Hippocampal encoding ──
                _episodic_mem_id = await self.memory.encode(
```

with:

```python
            try:
                # ── Expression mode: skip all memory writes ──
                if _interaction_mode == "expression":
                    return

                # ── Phase 4: Hippocampal encoding ──
                _episodic_mem_id = await self.memory.encode(
```

- [ ] **Step 5: Commit**

```bash
git add brain_agent/pipeline.py brain_agent/memory/manager.py
git commit -m "feat(pipeline): branch on interaction_mode, skip writes in expression, add timestamps to retrieval"
```

---

### Task 7: PSC — High-Confidence Preference Change Storage

**Files:**
- Modify: `brain_agent/pipeline.py`

- [ ] **Step 1: Update PSC system prompt for preference change detection**

In `_PSC_SYSTEM_PROMPT` (around line 284), after the existing story graph rules, before the closing `"` of the string, add:

```python
        "\n- PREFERENCE CHANGES are critical: if user reveals a change in preference, emotion, "
        "or attitude (e.g., 'I used to like X but now I don't'), mark confidence=1.0. "
        "This is a confirmed correction, not inference."
```

Append this line to the end of the `_PSC_SYSTEM_PROMPT` string concatenation.

- [ ] **Step 2: Commit**

```bash
git add brain_agent/pipeline.py
git commit -m "feat(psc): enforce high confidence for preference change detection"
```

---

### Task 8: Integration Test

**Files:**
- Manual testing via dashboard

- [ ] **Step 1: Start the server**

```bash
python -m brain_agent.dashboard.server
```

- [ ] **Step 2: Test Question mode**

1. Open dashboard in browser
2. Verify toggle shows "Question | Expression" above the lightbulb
3. Select "Question" (should be default, orange highlight)
4. Send a message: "나 오늘 짬뽕 먹었어"
5. Verify: response is normal, memory is updated (check LangFuse trace for PSC span)

- [ ] **Step 3: Test Expression mode**

1. Switch toggle to "Expression" (blue highlight)
2. Send: "내가 뭘 좋아해?"
3. Verify: response only mentions known facts from identity_facts
4. Send: "나 홍합 좋아해?"
5. Verify: if no preference stored, response says "모르겠어" without guessing
6. Check LangFuse: no PSC span, no encoding span

- [ ] **Step 4: Verify no memory writes in Expression mode**

1. Note current identity_facts count: `python -c "import sqlite3; print(sqlite3.connect('data/graph.db').execute('SELECT COUNT(*) FROM identity_facts').fetchone())"`
2. Send several messages in Expression mode
3. Verify count is unchanged

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for Question/Expression mode"
```
