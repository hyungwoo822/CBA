# Question / Expression Mode Design

**Date:** 2026-04-15
**Status:** Approved

---

## Overview

Brain agent operates in two interaction modes, toggled by the user via dashboard HUD:

- **Question mode** — Agent learns about the user. Detects logical gaps in memory (contradictions, missing causal links, unconfirmed preference changes) and asks about them mid-conversation. All memory write paths active.
- **Expression mode** — Agent expresses what it knows. Responds only from stored facts. No inference, no extrapolation, no guessing. All memory write paths disabled (read-only).

`interaction_mode` is orthogonal to `network_mode` (ECN/DMN/Creative). Network mode governs internal brain state; interaction mode governs user-facing behavior.

---

## Question Mode

### PFC Behavior

PFC system prompt receives a memory gap analysis instruction block when `interaction_mode == "question"`.

**Gap detection targets:**
1. **Time-based contradictions** — Facts that conflict across timestamps. E.g., "sick on 04-07" but "went hiking on 04-10". Requires timestamp in Retrieved Memories.
2. **Causal/preference inference** — Events with unconfirmed emotional or preference consequences. E.g., "hospitalized because of mussel" → mussel preference unknown.
3. **Emotional/rational outcome** — Events with no recorded resolution. E.g., "had a bad day" with no follow-up on how it resolved.

**Constraints:**
- No tail-chasing questions (don't continue a conversation the user ended).
- Maximum 1 question per turn.
- Only question when a gap exists. Otherwise, normal conversation.

### Memory Writes

All write paths active (same as current behavior):
- PSC entity extraction → identity_facts, knowledge_graph, ChromaDB
- Phase 4 episodic encoding
- Phase 5 consolidation, forgetting, dreaming

### Preference Change Detection

When a user's answer reveals a change (e.g., "used to like coffee, now I don't"), PSC stores this with **confidence 1.0** to overwrite the old fact. This is a confirmed correction, not inference.

### Timestamp in Retrieved Memories

Current format:
```
[semantic|rel=0.40] user witness heart attack
```

New format:
```
[semantic|rel=0.40|2026-04-07] user witness heart attack
```

Source: episodic store `timestamp` field and semantic store `last_seen` field.

---

## Expression Mode

### PFC Behavior

PFC system prompt is replaced with a strict fact-only instruction:
- Answer only from facts stored in memory.
- If a fact is not in memory, respond "I don't know that" or equivalent.
- No inference, no "maybe", no "probably", no extrapolation.
- No logical gap questions (that's Question mode's job).

### Memory Writes — ALL DISABLED

| Component | Action |
|---|---|
| Phase 4 encoding | Skip |
| PSC (entity extraction) | Skip |
| Phase 5 consolidation | Skip |
| Ebbinghaus forgetting | Skip |
| Homeostatic scaling | Skip |
| Pruning | Skip |
| Dreaming promotion | Skip |
| Retroactive interference | Skip |
| identity_facts writes | Skip |
| knowledge_graph writes | Skip |
| ChromaDB writes | Skip |

### Memory Reads — ACTIVE

| Component | Action |
|---|---|
| Phase 6 retrieval | Normal |
| identity_facts reads | Normal |
| knowledge_graph reads | Normal |
| ChromaDB search | Normal |
| spread_activation | Normal |

---

## UI: Dashboard Toggle

### Placement

Positioned directly above the existing lightbulb icon (Knowledge Graph toggle):
- Lightbulb: `position: fixed; top: 18%; left: 50%`
- Mode toggle: `position: fixed; top: 13%; left: 50%; transform: translateX(-50%)`

### Design

Pill-shaped segment toggle:
```
┌──────────┬────────────┐
│ Question │ Expression │
└──────────┴────────────┘
```

- **Question selected**: orange highlight (`#f97316`), subtle glow
- **Expression selected**: blue highlight (`#60a5fa`), subtle glow
- Unselected segment: translucent dark (`rgba(15,23,42,0.6)`)
- Border: `1px solid rgba(148,163,184,0.15)`
- Font: 12px, same as HUD elements
- Backdrop blur for glass effect

### State

- Zustand store: `interactionMode: 'question' | 'expression'`
- Action: `setInteractionMode(mode)`
- Default: `'question'`

---

## API Changes

### POST /api/process

Add optional `mode` field to form data:

```
FormData:
  text: "내가 뭘 좋아해?"
  mode: "expression"        ← new, defaults to "question"
```

Server extracts `mode` and passes to `BrainAgent.process()`.

---

## Pipeline Changes

### process_request signature

```python
async def process_request(
    self, text: str = "", image: bytes | None = None,
    audio: bytes | None = None, trace_run=None,
    interaction_mode: str = "question",    # ← new
) -> PipelineResult:
```

### Branching points

1. **PFC system prompt** (Phase 7): `build_cortical_system_prompt()` receives `interaction_mode` and appends mode-specific instruction block.
2. **Background post-response**: When `interaction_mode == "expression"`, skip the entire `_background_post_response()` function (no encoding, no PSC, no consolidation, no forgetting, no dreaming).

---

## Files Changed

| File | Change |
|---|---|
| `dashboard/src/stores/brainState.ts` | Add `interactionMode`, `setInteractionMode`, include in `submitChat` |
| `dashboard/src/components/InteractionModeToggle.tsx` | New component: pill toggle above lightbulb |
| `dashboard/src/App.tsx` | Mount InteractionModeToggle |
| `dashboard/src/styles/index.css` | Toggle styles |
| `brain_agent/dashboard/server.py` | Extract `mode` from form data, pass to agent |
| `brain_agent/agent.py` | Forward `mode` to pipeline |
| `brain_agent/pipeline.py` | `interaction_mode` parameter, skip writes in expression mode |
| `brain_agent/regions/prefrontal.py` | Mode-specific prompt blocks in `build_cortical_system_prompt()` |
