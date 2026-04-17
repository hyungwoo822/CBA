"""Multi-stage adaptive extractor.

The package implements Triage, Extract, Temporal Resolve, Validate, Severity,
Refine, and staging-only persistence. All LLM calls go through
brain_agent.providers.base.LLMProvider.
"""
from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.orchestrator import ExtractionOrchestrator
from brain_agent.extraction.types import (
    ExtractionResult,
    SeverityDecision,
    TemporalResolveResult,
    TriageResult,
    ValidationResult,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionOrchestrator",
    "ExtractionResult",
    "SeverityDecision",
    "TemporalResolveResult",
    "TriageResult",
    "ValidationResult",
]
