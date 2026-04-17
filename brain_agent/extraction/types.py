"""Dataclass contracts for the multi-stage extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TriageResult:
    """Output of Stage 1 Triage."""

    target_workspace_id: str
    input_kinds: list[str]
    severity_hint: str = "none"
    skip_stages: list[int] = field(default_factory=list)
    workspace_ask: str | None = None


@dataclass
class TemporalResolveResult:
    """Output of Stage 2.5 Temporal Resolve (C3)."""

    update_ops: list[dict] = field(default_factory=list)
    new_edges: list[dict] = field(default_factory=list)
    reinforced_edges: list[dict] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Output of Stage 3 Validate."""

    contradictions: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)


@dataclass
class SeverityDecision:
    """Output of Stage 4 severity branch."""

    response_mode: str = "normal"
    clarification_questions: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Top-level return value of ExtractionOrchestrator.extract()."""

    workspace_id: str
    source_id: str
    narrative_chunk: str
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)
    new_type_proposals: list[dict] = field(default_factory=list)
    response_text: str = ""
    response_mode: str = "normal"
    clarification_questions: list[str] = field(default_factory=list)
    workspace_ask: str | None = None
