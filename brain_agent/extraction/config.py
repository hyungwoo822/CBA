"""Configuration for the multi-stage extraction pipeline.

Model fields are string identifiers. ``"auto"`` means each stage asks the
provided LLMProvider for its default model at call time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    triage_model: str = "auto"
    extract_model: str = "auto"
    temporal_classify_model: str = "auto"
    refine_model: str = "auto"

    max_retry: int = 1
    enable_severity_block: bool = True
    promotion_threshold_n: int = 3

    max_open_questions_per_extraction: int = 3
    fok_similarity_threshold: float = 0.3
    pattern_separation_label_similarity_threshold: float = 0.75
    pattern_separation_window_hours: int = 24
