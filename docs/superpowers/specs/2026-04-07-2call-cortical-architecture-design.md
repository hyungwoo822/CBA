# 2-Call Cortical Architecture

> 7 LLM calls per cycle → 2 calls. ~60-70% token reduction.
> Neuroscience justification: Pessoa 2008, Diekelmann & Born 2010.

## Problem

Current pipeline makes 7 LLM calls per request cycle (~10K-17K tokens):
- Wernicke (comprehension), Amygdala R+L (appraisal), PFC (planning)
- Background: Broca (refinement), user_facts extraction, procedural caching

## Solution

Merge into 2 calls: **Cortical Integration** (foreground) + **Post-Synaptic Consolidation** (background).

### Architecture

```
Input
  │
  ▼
[Algorithmic Pre-processing] ── LLM 0 calls
  Thalamus: modality detection
  Structural keyword extraction (extended _structural_parse)
  memory.retrieve(text): embedding-based memory search
  memory.retrieve_identity(): DB load self/user model
  memory.procedural.match(text): procedural cache check
  neuromodulators.snapshot(): current NT state
  │
  ▼
[Foreground LLM] ── 1 call, ~3,000 tokens
  Cortical Integration (Wernicke + Amygdala + PFC unified)
  Input: identity + memories + NT state + tool defs + user text
  Output JSON:
    comprehension: {intent, complexity, keywords, language}
    appraisal: {valence, arousal, threat_detected, primary_emotion}
    plan: {actions: [...], confidence}
    entities: {about_user: [...], knowledge: [...]}
  │
  ▼
[Result Distribution] ── inject into existing regions
  wernicke.inject(comprehension) → activation_level set
  amygdala.inject(appraisal) → emotional_tag set
  plan → BasalGanglia → Cerebellum → Tool loop (unchanged)
  entities → _store_identity_facts_realtime()
  │
  ▼
[Response returned]
  │
  ▼
[Background LLM] ── 1 call, ~1,800 tokens
  Post-Synaptic Consolidation (Broca + facts + procedural unified)
  Input: original text + agent response + comprehension context
  Output JSON:
    refined_response: string | null  → WS broadcast if different
    user_facts: {entities, relations} → semantic store
    procedural: {trigger, strategy} | null → procedural cache
```

### Token Budget

| Call | Prompt | Completion | Total |
|------|--------|-----------|-------|
| Cortical Integration (FG) | ~1,500 | ~1,500 | ~3,000 |
| PSC (BG) | ~1,000 | ~800 | ~1,800 |
| **Total** | | | **~4,800** |

Tool usage adds N extra calls (PFC re-plan per tool result).

### Region Changes

**Modified (inject pattern):**
- `wernicke.py`: Add `inject(comprehension)`, remove direct LLM call from `process()`
- `amygdala.py`: Add `inject(appraisal)`, remove direct LLM calls from R/L
- `broca.py`: Remove direct LLM call, receive refined_response from PSC
- `prefrontal.py`: System prompt absorbs comprehension+appraisal context

**Unchanged:**
- All algorithmic regions (Thalamus, BasalGanglia, Cerebellum, ACC, Insula, etc.)
- ToolRegistry, MCP, Middleware (barrier/myelin/meninges)
- Dashboard 3D visualization (activation_level preserved)
- Memory system, Neuromodulator system

### Pipeline.process_request() New Flow

```python
# Phase 1: Algorithmic pre-processing
thalamus_signal = await self.thalamus.process(input_signal)
keywords = self._structural_keyword_extract(text)
memories = await self.memory.retrieve(text, top_k=5)
identity = await self.memory.retrieve_identity()
cached = await self.memory.procedural.match(text)
nt_state = self.neuromodulators.snapshot()

# Phase 2: Single foreground LLM call
cortical_result = await self._cortical_integration(
    text, keywords, memories, identity, nt_state, cached
)

# Phase 3: Distribute to regions
self.wernicke.inject(cortical_result["comprehension"])
self.amygdala.inject(cortical_result["appraisal"])
plan_signal = self._build_plan_signal(cortical_result["plan"])

# Phase 4: Executive processing (unchanged)
# ACC → BasalGanglia → Cerebellum → Tool loop → response

# Phase 5: Background PSC (non-blocking)
asyncio.create_task(self._post_synaptic_consolidation(...))
```

### Neuroscience Justification

- **Merging Wernicke+Amygdala**: Pessoa 2008 — cognition and emotion are integrated, not segregated. Lindquist et al. 2012 — shared neural substrate.
- **Merging background calls**: Diekelmann & Born 2010 — hippocampal replay consolidates declarative + procedural memory in a single cycle.
- **Algorithmic pre-processing as subcortical**: Thalamus, brainstem, basal ganglia operate algorithmically in the real brain — no "LLM equivalent" needed.
- **Region injection**: Regions remain as anatomical units with activation levels, matching the brain's modular anatomy even when processing is integrated.
