"""Stage 1: multi-label triage with workspace override detection."""
from __future__ import annotations

import re
from typing import Any

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.types import TriageResult
from brain_agent.providers.base import LLMProvider

_GREETING_PATTERNS = [r"\b(hi|hello|hey|yo)\b", r"안녕", r"반가"]
_FAREWELL_PATTERNS = [r"\b(bye|goodbye|see\s+ya|cya|later)\b", r"잘\s*있어", r"갈게"]
_CONFIRMATION_PATTERNS = [r"\b(yes|yep|yeah|ok|okay|sure|right|correct)\b", r"맞아", r"맞다", r"그래"]
_QUESTION_PATTERNS = [
    r"\?$",
    r"\?\s",
    r"\b(what|why|how|when|where|who)\b",
    r"뭐야",
    r"어떻게",
    r"언제",
    r"어디",
    r"괜찮",
    r"할까",
    r"지\?",
]
_CORRECTION_PATTERNS = [r"\b(actually|wait|hold\s+on|correction)\b", r"수정", r"바꾸", r"아니", r"정정"]
_REQUEST_PATTERNS = [r"\b(please|pls|can\s+you|could\s+you|would\s+you)\b", r"해줘", r"부탁", r"줘"]
_SPEC_DROP_PATTERNS = [
    r"\b(spec|requirement|api|endpoint|contract|schema|rate\s*limit|idempoten|retry|flow)\b",
    r"정책",
    r"규칙",
    r"제약",
    r"스펙",
    r"요구사항",
]

_PATTERN_TABLE: list[tuple[str, list[str]]] = [
    ("greeting", _GREETING_PATTERNS),
    ("farewell", _FAREWELL_PATTERNS),
    ("confirmation", _CONFIRMATION_PATTERNS),
    ("question", _QUESTION_PATTERNS),
    ("correction", _CORRECTION_PATTERNS),
    ("request", _REQUEST_PATTERNS),
    ("spec_drop", _SPEC_DROP_PATTERNS),
]

_CONSERVATIVE_KINDS = {"correction", "spec_drop", "request"}
_STAGE_2_SKIP_OK = {"greeting", "farewell", "confirmation", "question"}
_STAGE_3_SKIP_OK = {"greeting", "farewell", "confirmation"}


def _match_labels(text: str) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []
    for label, patterns in _PATTERN_TABLE:
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            labels.append(label)
    return labels


def _compute_skip_stages(input_kinds: list[str]) -> list[int]:
    if any(kind in _CONSERVATIVE_KINDS for kind in input_kinds):
        return []
    kind_set = set(input_kinds)
    skip: list[int] = []
    if kind_set and kind_set.issubset(_STAGE_2_SKIP_OK):
        skip.append(2)
    if kind_set and kind_set.issubset(_STAGE_3_SKIP_OK):
        skip.append(3)
    return skip


class Triage:
    def __init__(
        self,
        workspace_store,
        llm_provider: LLMProvider,
        config: ExtractionConfig,
    ):
        self._ws = workspace_store
        self._llm = llm_provider
        self._cfg = config

    async def classify(
        self,
        text: str,
        session_id: str,
        comprehension: dict[str, Any] | None,
    ) -> TriageResult:
        current_ws = await self._ws.get_session_workspace(session_id) or "personal"

        kinds = _match_labels(text)
        if not kinds:
            kinds = ["spec_drop"]
        skip_stages = _compute_skip_stages(kinds)

        workspace_ask: str | None = None
        if comprehension and comprehension.get("workspace_hint"):
            hint = str(comprehension["workspace_hint"])
            hint_conf = float(
                comprehension.get(
                    "workspace_hint_confidence",
                    comprehension.get("confidence", 0.0),
                )
            )
            if hint_conf > 0.8 and hint != current_ws:
                candidate = await self._ws.get_workspace(hint)
                if candidate is not None:
                    workspace_ask = (
                        f"This looks like it belongs in the "
                        f"{candidate.get('name', hint)} workspace. "
                        f"Should I use that instead of {current_ws}?"
                    )

        return TriageResult(
            target_workspace_id=current_ws,
            input_kinds=kinds,
            severity_hint="none",
            skip_stages=skip_stages,
            workspace_ask=workspace_ask,
        )
