"""Stage 4: map validation output to a response mode."""
from __future__ import annotations

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.types import SeverityDecision


def compute_response_mode(
    contradictions: list[dict],
    open_questions: list[dict],
    config: ExtractionConfig,
) -> SeverityDecision:
    has_severe = any(c.get("severity") == "severe" for c in contradictions) or any(
        q.get("severity") == "severe" for q in open_questions
    )
    has_moderate_issue = bool(contradictions) or any(
        q.get("severity") == "moderate" for q in open_questions
    )

    if has_severe and config.enable_severity_block:
        mode = "block"
    elif has_severe or has_moderate_issue:
        mode = "append"
    else:
        mode = "normal"

    return SeverityDecision(
        response_mode=mode,
        clarification_questions=[q["question"] for q in open_questions],
    )
