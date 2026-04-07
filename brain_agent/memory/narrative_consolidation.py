"""Narrative Memory Consolidation — LLM-driven identity_facts + SOUL.md updates.

During Phase 5 (Consolidation), this system:
1. Reads recent episodic memories from hippocampal staging
2. Asks an LLM to extract identity facts and agent personality updates
3. Stores identity_facts in the knowledge graph (single source of truth)
4. Updates SOUL.md (agent self-schema — not representable as K-V)

identity_facts (user_model + self_model) replace the former MEMORY.md and
USER.md files as the single source of truth for user/agent knowledge.
PFC and TPJ render these facts into prompt context at query time.

References:
  - McClelland et al. (1995): Complementary Learning Systems
  - Winocur & Moscovitch (2011): Episodic→semantic transformation
"""
from __future__ import annotations

import json
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


_CONSOLIDATION_SYSTEM_PROMPT = """\
You are the neocortical consolidation system of a neural agent's brain.
You process recent conversation memories and extract structured identity facts.

## What you produce:

1. **identity_facts** — Structured key-value pairs stored in the knowledge graph.
   This is the SINGLE SOURCE OF TRUTH for all user and agent knowledge.

2. **soul_update** — Updated SOUL.md (agent's evolving self-schema).

## identity_facts rules:

### user_model (facts about the user):
Extract CONCRETE facts using structured keys:
- Use category prefixes for grouping: `preference:X`, `attribute:X`, `habit:X`, `social:X`
- Top-level keys for core profile: `name`, `language`, `timezone`, `occupation`, `personality_traits`
- AGGRESSIVELY INFER — even weak inference (marked 추정) is better than missing
- Strip Korean particles from names (야/이야/는/은/가/이)
- Each fact has: key, value, confidence (0.0-1.0)
- If new info CONTRADICTS existing facts, output the UPDATED value (UPSERT semantics)
- Do NOT store trivial small talk ("said hello", "said goodnight")

### self_model (facts about the agent):
Extract traits the agent is developing:
- `communication_style`, `knowledge_areas`, `humor_style`, `relationship_dynamic`
- Only include traits actually demonstrated in conversations

## soul_update rules:
SOUL.md is the agent's self-concept — who it is, how it relates to this user.
- PRESERVE base structure (Core Identity, Personality, Values)
- EVOLVE content based on actual conversations
- Add/update **## Our Relationship** and **## What I've Learned** sections
- Keep authentic — only claim traits actually shown
- Max ~40 lines. Concise and real > long and generic.

## Output format:

Return a JSON object:
{
  "soul_update": "# Soul — Neural Identity\\n\\n...",
  "identity_facts": {
    "self_model": [{"key": "...", "value": "...", "confidence": 0.9}],
    "user_model": [{"key": "...", "value": "...", "confidence": 0.9}]
  }
}

Return ONLY valid JSON. No markdown fences, no explanation."""


async def narrative_consolidate(
    memories: list[dict],
    llm_provider: LLMProvider | None = None,
    semantic_store=None,
) -> bool:
    """Run LLM-driven consolidation on recent memories.

    Updates identity_facts in the knowledge graph (single source of truth)
    and SOUL.md (agent self-schema).

    Returns True if consolidation succeeded.
    """
    if not memories:
        logger.warning("Narrative consolidation skipped: no staging memories")
        return False
    if not llm_provider:
        logger.error("Narrative consolidation BLOCKED: llm_provider is None.")
        return False

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

    # Load current identity facts for cross-reference
    identity_section = ""
    if semantic_store:
        try:
            user_facts = await semantic_store.get_identity_facts("user_model")
            self_facts = await semantic_store.get_identity_facts("self_model")
            fact_lines = []
            if user_facts:
                fact_lines.append("### Current User Model")
                for f in user_facts:
                    fact_lines.append(f"- {f['key']}: {f['value']} (conf={f['confidence']})")
            if self_facts:
                fact_lines.append("### Current Self Model")
                for f in self_facts:
                    fact_lines.append(f"- {f['key']}: {f['value']}")
            if fact_lines:
                identity_section = (
                    "\n## Current Identity Facts (update/add as needed — recent memories OVERRIDE)\n"
                    + "\n".join(fact_lines)
                )
        except Exception:
            pass

    prompt = f"""## Current SOUL.md
{current_soul or "(empty — first consolidation)"}
{identity_section}

## Recent Conversation Memories to Consolidate
{chr(10).join(f"- {line}" for line in memory_lines)}"""

    try:
        response = await llm_provider.chat(
            messages=[
                {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.2,
        )

        if not response.content:
            logger.error("Narrative consolidation: LLM returned empty content.")
            return False

        # Parse JSON response
        text = response.content.strip()
        logger.info("Narrative consolidation: LLM response length=%d", len(text))

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            try:
                fixed = re.sub(r'(?<=": ")(.*?)(?="[,\}])',
                               lambda m: m.group(0).replace('\n', '\\n'),
                               text, flags=re.DOTALL)
                data = json.loads(fixed)
            except (json.JSONDecodeError, Exception):
                pass

        if data is None:
            logger.error("Narrative consolidation: JSON parse failed. Preview: %s", text[:500])
            return False

        logger.info("Narrative consolidation: parsed keys=%s", list(data.keys()))

        # Update SOUL.md
        soul_update = data.get("soul_update", "")
        if isinstance(soul_update, dict):
            soul_update = json.dumps(soul_update, ensure_ascii=False, indent=2)
        if soul_update and soul_update != current_soul:
            _write_file("SOUL.md", soul_update)
            logger.info("Narrative consolidation: SOUL.md updated")

        # Store identity facts (single source of truth)
        identity_facts = data.get("identity_facts", {})
        facts_stored = 0
        if semantic_store and identity_facts:
            for fact in identity_facts.get("self_model", []):
                if fact.get("key") and fact.get("value"):
                    conf = fact.get("confidence", 1.0)
                    await semantic_store.add_identity_fact(
                        "self_model", fact["key"], fact["value"],
                        source="consolidation", confidence=conf,
                    )
                    facts_stored += 1
            for fact in identity_facts.get("user_model", []):
                if fact.get("key") and fact.get("value"):
                    conf = fact.get("confidence", 1.0)
                    await semantic_store.add_identity_fact(
                        "user_model", fact["key"], fact["value"],
                        source="consolidation", confidence=conf,
                    )
                    facts_stored += 1
            logger.info("Narrative consolidation: stored %d identity facts", facts_stored)

        return True

    except Exception as e:
        logger.warning("Narrative consolidation failed: %s", e)
        return False
