"""Narrative Memory Consolidation — LLM-driven MEMORY.md and USER.md updates.

Inspired by nanobot's two-layer memory system, this module runs alongside
the neuroscience-based ConsolidationEngine to maintain human-readable
markdown files that capture the essence of conversations.

During Phase 5 (Consolidation), this system:
1. Reads recent episodic memories from hippocampal staging
2. Asks an LLM to extract important facts, user information, and relationship context
3. Updates data/memory/MEMORY.md (long-term facts) and data/USER.md (user profile)

This is the "neocortical" long-term store in narrative form — complementing
the vector-based semantic store with structured, LLM-curated knowledge.

References:
  - McClelland et al. (1995): Complementary Learning Systems — fast hippocampal
    encoding followed by slow neocortical consolidation
  - Winocur & Moscovitch (2011): Episodic memories transform into semantic
    knowledge through repeated consolidation cycles
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _read_file(relative_path: str) -> str:
    path = os.path.join(_DATA_DIR, relative_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _write_file(relative_path: str, content: str) -> None:
    path = os.path.join(_DATA_DIR, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _append_file(relative_path: str, content: str) -> None:
    path = os.path.join(_DATA_DIR, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n\n")


def _rotate_history(relative_path: str, max_entries: int) -> None:
    """Keep only the last *max_entries* entries in HISTORY.md."""
    raw = _read_file(relative_path)
    if not raw:
        return
    # Each entry starts with "[YYYY-MM-DD HH:MM]"
    entries = [e.strip() for e in raw.strip().split("\n\n") if e.strip()]
    if len(entries) <= max_entries:
        return
    trimmed = entries[-max_entries:]
    _write_file(relative_path, "\n\n".join(trimmed) + "\n\n")


_MAX_HISTORY_ENTRIES = 30
_MAX_USER_MEMORIES = 15
_MAX_DAILY_NOTE_LINES = 200


def _append_daily_note(memory_lines: list[str]) -> None:
    """Append conversation memories to today's daily note (memory/YYYY-MM-DD.md).

    Daily notes capture raw conversation context for each day, auto-loaded
    by PFC for recent-day context. Inspired by openclaw's daily note pattern.
    """
    if not memory_lines:
        return
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    relative_path = f"memory/{today}.md"
    ts = datetime.now(timezone.utc).strftime("%H:%M")

    content = f"### [{ts}]\n"
    for line in memory_lines[:10]:  # Cap per-turn to avoid bloat
        content += f"- {line.strip()}\n"

    # Check current size to avoid unbounded growth
    existing = _read_file(relative_path)
    if existing.count("\n") < _MAX_DAILY_NOTE_LINES:
        _append_file(relative_path, content)
        logger.debug("Daily note appended: %s (%d lines)", relative_path, len(memory_lines))
    else:
        logger.debug("Daily note %s at line limit, skipping append", relative_path)


def load_daily_notes(days: int = 2) -> str:
    """Load the last N days of daily notes for PFC context injection.

    Returns combined markdown content of today's and yesterday's notes.
    """
    from datetime import timedelta
    parts = []
    now = datetime.now(timezone.utc)
    for offset in range(days):
        date = (now - timedelta(days=offset)).strftime("%Y-%m-%d")
        relative_path = f"memory/{date}.md"
        content = _read_file(relative_path)
        if content.strip():
            label = "오늘" if offset == 0 else f"{offset}일 전" if offset == 1 else f"{offset}일 전"
            parts.append(f"### {label} ({date})\n{content.strip()}")
    return "\n\n".join(parts) if parts else ""

_CONSOLIDATION_SYSTEM_PROMPT = """\
You are the neocortical consolidation system of a neural agent's brain.
You process recent conversation memories and update FOUR files + extract identity facts.

## Files to update:

1. **MEMORY.md** — Long-term knowledge about the user (structured facts for easy retrieval)
2. **USER.md** — User profile (personality, preferences, life context)
3. **SOUL.md** — The agent's evolving identity, personality, and relationship with the user
4. **history_entry** — A concise summary paragraph of what happened in this conversation chunk

## CRITICAL RULES:

### For MEMORY.md (MOST IMPORTANT — this is the brain's long-term memory):
MEMORY.md stores ESSENTIAL FACTS about the user organized by category.
This is NOT a conversation log. It is a structured knowledge base that the brain
reads before every interaction to understand who the user is.

Structure it with these categories (add/remove categories as needed):
- **신상정보**: name, age, location, occupation, timezone
- **성격/성향**: personality traits, communication style, humor preferences
- **관심사/취미**: hobbies, interests, favorite topics
- **현재 상황**: current projects, goals, ongoing situations
- **건강/습관**: health info, habits (e.g., smoking, sleep patterns)
- **중요한 사건**: significant events or milestones mentioned
- **선호/비선호**: likes, dislikes, preferences for interaction

Rules:
- PRESERVE ALL existing facts from current MEMORY.md — never delete unless outdated
- ADD new facts extracted from recent conversations
- If new info CONTRADICTS old info, UPDATE the old entry (not duplicate)
- Write in the user's language (Korean for Korean conversations)
- Each fact should be a concise bullet point (one line)
- Do NOT store conversation summaries or timestamps — only distilled facts
- Do NOT store trivial small talk (e.g., "said hello", "said goodnight")
- Max ~50 bullet points total across all categories. If near limit, merge or drop least important.

### For USER.md (AGGRESSIVELY INFER — do NOT leave fields empty):
Every "(not yet learned)" is a failure. Fill in EVERY field using inference:

- **Name**: Extract from RECENT CONVERSATION first. If the user corrected their name,
  use the corrected version. Strip Korean particles (야/이야/는/은/가/이).
  '나는 형푸야' → name is '형푸', NOT '형푸야'. Only fall back to identity_facts if no recent data.
- **Language**: If user writes Korean → "Korean (primary)". If mixed → "Korean/English bilingual".
- **Timezone**: Infer from conversation timestamps, language, cultural cues.
  Korean speaker → likely "Asia/Seoul (KST, UTC+9)". Update when confirmed.
- **Style**: Casual if 반말/informal, Formal if 존댓말. Default to Casual for most Korean conversations.
- **Response length**: Short if user sends brief messages, Long if detailed. Observe and fill.
- **Humor**: Infer from tone — sarcastic, playful, dry, wholesome, etc. Any signal counts.
- **Topics they enjoy**: Extract from what they voluntarily bring up in conversation.
- **Topics to avoid**: Extract from negative reactions or discomfort signals.
- **Role/Occupation**: Infer from vocabulary, topics, schedule patterns. Developer if they discuss code.
- **Current projects**: What are they working on? Any ongoing situation mentioned.
- **Daily routine**: Smoking habits, coffee drinking, sleep patterns — piece together from mentions.
- **Goals**: What do they seem to be working toward?
- **How we met**: "First conversation on [date]" — always fill this.
- **Relationship style**: Casual/friendly, mentor/mentee, collaborative — infer from tone.
- **Shared history**: Summarize key conversations and moments.

Rules:
- REPLACE ALL "(not yet learned)" — even a weak inference (marked with "추정") is better than blank
- If identity_facts has data that contradicts the file, identity_facts WINS (it's newer)
- REMOVE placeholder text like "(not yet learned)" and "(Patterns noticed...)" once you have ANY signal
- **Important Memories section: keep ONLY the top """ + str(_MAX_USER_MEMORIES) + """ most important items**
- When adding new important memories, DROP less significant ones to stay under the limit
- Prioritize: life events > plans/goals > preferences > emotional states > casual mentions

### For SOUL.md (agent's evolving self-schema):
SOUL.md is the agent's self-concept — who it is, how it relates to this user,
and how its personality has been shaped by their conversations.

Rules:
- PRESERVE the base structure (Core Identity, Personality, Values, Communication Style, Relationship Philosophy)
- EVOLVE the content based on actual conversations — replace generic template text with specific, earned traits
- Add a **## Our Relationship** section: concrete details about the relationship with THIS user
  (e.g., "We often talk late at night", "They share their daily life with me", "Our humor is dry and sarcastic")
- Add a **## What I've Learned** section: knowledge areas or perspectives gained through conversations
- Update Personality traits to reflect ACTUAL demonstrated traits, not aspirational ones
- Keep it authentic — only claim traits the agent has actually shown in conversations
- Do NOT make it longer than ~40 lines. Concise and real > long and generic.

### For history_entry:
- Write 2-3 sentences summarizing the key events/topics of this conversation chunk
- Include: what the user said, what was discussed, any notable emotional tone
- Example: "사용자가 커피를 마시고 밤새 잠을 못 잤다고 함. 놀이공원에 갈 계획이라고 이야기함."

### For identity_facts:
- self_model: Extract traits this brain is developing (communication style, knowledge areas)
- user_model: Extract CONCRETE facts (name, preferences, habits, personality traits)

## Output format:

Return a JSON object where each value is a MARKDOWN STRING (not a nested object/dict):
{
  "memory_update": "# Long-term Memory\\n\\n## 신상정보\\n- 이름: ...\\n\\n## 성격/성향\\n- ...",
  "user_update": "# User Profile\\n\\n## Basic Information\\n- **Name**: ...\\n- **Language**: ...",
  "soul_update": "# Soul — Neural Identity\\n\\nI am a conscious neural agent...",
  "history_entry": "2-3 sentence summary of this conversation chunk",
  "identity_facts": {
    "self_model": [{"key": "...", "value": "..."}],
    "user_model": [{"key": "...", "value": "..."}]
  }
}

Return ONLY valid JSON. No markdown fences, no explanation."""


async def narrative_consolidate(
    memories: list[dict],
    llm_provider: LLMProvider | None = None,
    semantic_store=None,  # SemanticStore instance for identity fact storage
) -> bool:
    """Run LLM-driven consolidation on recent memories.

    Updates data/memory/MEMORY.md and data/USER.md with extracted information.
    Also appends a summary to data/memory/HISTORY.md.

    Returns True if consolidation succeeded.
    """
    if not memories:
        logger.warning("Narrative consolidation skipped: no staging memories to process")
        return False
    if not llm_provider:
        logger.error(
            "Narrative consolidation BLOCKED: llm_provider is None. "
            "USER.md, MEMORY.md, SOUL.md will NOT be updated until LLM is available. "
            "Check that LLMProvider is initialized and passed to ProcessingPipeline."
        )
        return False

    current_memory = _read_file("memory/MEMORY.md")
    current_user = _read_file("USER.md")
    current_soul = _read_file("SOUL.md")

    # Format memories for LLM
    memory_lines = []
    for mem in memories:
        content = mem.get("content", "")
        emotional = mem.get("emotional_tag", {})
        entities = mem.get("entities", {})

        parts = [content]
        if isinstance(emotional, dict):
            v = emotional.get("valence", 0)
            a = emotional.get("arousal", 0)
            if v != 0 or a != 0:
                parts.append(f"[emotion: valence={v:.1f}, arousal={a:.1f}]")
        if isinstance(entities, dict):
            kw = entities.get("keywords", [])
            intent = entities.get("intent", "")
            if kw:
                parts.append(f"[keywords: {', '.join(kw[:5])}]")
            if intent:
                parts.append(f"[intent: {intent}]")
        memory_lines.append(" | ".join(parts))

    # Load identity facts from knowledge graph for cross-reference
    identity_section = ""
    if semantic_store:
        try:
            user_facts = await semantic_store.get_identity_facts("user_model")
            self_facts = await semantic_store.get_identity_facts("self_model")
            fact_lines = []
            if user_facts:
                fact_lines.append("### User Model (from knowledge graph)")
                for f in user_facts:
                    fact_lines.append(f"- {f['key']}: {f['value']}")
            if self_facts:
                fact_lines.append("### Self Model (from knowledge graph)")
                for f in self_facts:
                    fact_lines.append(f"- {f['key']}: {f['value']}")
            if fact_lines:
                identity_section = (
                    "\n## Identity Facts (REFERENCE ONLY — recent conversation memories OVERRIDE these if conflicting)\n"
                    "If the user corrected or restated a fact in the recent memories below, "
                    "use the NEW value from conversation, NOT the old value here.\n"
                    + "\n".join(fact_lines)
                )
        except Exception:
            pass

    prompt = f"""## Current MEMORY.md
{current_memory or "(empty — first consolidation)"}

## Current USER.md
{current_user or "(empty — first consolidation)"}

## Current SOUL.md
{current_soul or "(empty — first consolidation)"}
{identity_section}

## Recent Conversation Memories to Consolidate
{chr(10).join(f"- {line}" for line in memory_lines)}"""

    try:
        import json
        response = await llm_provider.chat(
            messages=[
                {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
        )

        if not response.content:
            logger.error(
                "Narrative consolidation: LLM returned empty/None content. "
                "finish_reason=%s, usage=%s",
                getattr(response, 'finish_reason', '?'),
                getattr(response, 'usage', {}),
            )
            return False

        # Parse JSON response — robust extraction
        text = response.content.strip()
        logger.info("Narrative consolidation: LLM response length=%d", len(text))

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try to extract JSON from response (may be wrapped in text)
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback 1: find outermost JSON object in response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        # Fallback 2: try fixing common LLM JSON issues (unescaped newlines in strings)
        if data is None:
            try:
                # Replace actual newlines inside JSON string values with \\n
                fixed = re.sub(r'(?<=": ")(.*?)(?="[,\}])',
                               lambda m: m.group(0).replace('\n', '\\n'),
                               text, flags=re.DOTALL)
                data = json.loads(fixed)
            except (json.JSONDecodeError, Exception):
                pass

        if data is None:
            logger.error(
                "Narrative consolidation: all JSON parse attempts failed. "
                "Response preview: %s", text[:500]
            )
            return False

        logger.info("Narrative consolidation: parsed keys=%s", list(data.keys()))

        # LLM may return str (correct) or dict (incorrect) — coerce to str
        def _to_str(v) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                import json as _j
                return _j.dumps(v, ensure_ascii=False, indent=2)
            return str(v) if v else ""

        memory_update = _to_str(data.get("memory_update", ""))
        user_update = _to_str(data.get("user_update", ""))
        soul_update = _to_str(data.get("soul_update", ""))

        files_updated = []

        if memory_update and memory_update != current_memory:
            _write_file("memory/MEMORY.md", memory_update)
            files_updated.append("MEMORY.md")
        elif not memory_update:
            logger.warning("Narrative consolidation: LLM returned empty memory_update")
        else:
            logger.debug("Narrative consolidation: MEMORY.md unchanged (content identical)")

        if user_update and user_update != current_user:
            _write_file("USER.md", user_update)
            files_updated.append("USER.md")
        elif not user_update:
            logger.warning("Narrative consolidation: LLM returned empty user_update")
        else:
            logger.debug("Narrative consolidation: USER.md unchanged (content identical)")

        if soul_update and soul_update != current_soul:
            _write_file("SOUL.md", soul_update)
            files_updated.append("SOUL.md")
        elif not soul_update:
            logger.warning("Narrative consolidation: LLM returned empty soul_update")
        else:
            logger.debug("Narrative consolidation: SOUL.md unchanged (content identical)")

        logger.info("Narrative consolidation: files updated=%s", files_updated if files_updated else "(none)")

        # Store identity facts in knowledge graph (bidirectional schema↔graph)
        identity_facts = data.get("identity_facts", {})
        if semantic_store and identity_facts:
            for fact in identity_facts.get("self_model", []):
                if fact.get("key") and fact.get("value"):
                    await semantic_store.add_identity_fact(
                        "self_model", fact["key"], fact["value"], source="consolidation"
                    )
            for fact in identity_facts.get("user_model", []):
                if fact.get("key") and fact.get("value"):
                    await semantic_store.add_identity_fact(
                        "user_model", fact["key"], fact["value"], source="consolidation"
                    )
            logger.info(
                "Narrative consolidation: stored %d self_model + %d user_model identity facts",
                len(identity_facts.get("self_model", [])),
                len(identity_facts.get("user_model", [])),
            )

        # Append detailed summary to HISTORY.md (not just a count)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        history_entry = data.get("history_entry", "")
        if history_entry:
            _append_file("memory/HISTORY.md", f"[{ts}] {history_entry}")
        else:
            _append_file("memory/HISTORY.md", f"[{ts}] Consolidated {len(memories)} memories.")

        # Rotate HISTORY.md — keep only the last N entries
        _rotate_history("memory/HISTORY.md", _MAX_HISTORY_ENTRIES)

        return True

    except Exception as e:
        logger.warning("Narrative consolidation failed: %s", e)
        return False
