"""Content-driven activation profiling.

Computes per-region activation gains based on input characteristics,
so that emotional inputs strongly activate amygdala/insula while
analytical inputs strongly activate PFC/ACC.

References:
  - Pessoa (2008): The relationship between emotion and cognition
  - Duncan (2010): Multiple-demand system in the brain
  - Barrett & Satpute (2013): Large-scale brain networks in affective processing
"""
from __future__ import annotations


# Region groups for activation routing
# Names match the actual BrainRegion.name values in this codebase
EMOTIONAL_REGIONS = {"amygdala", "insula", "medial_pfc", "vta", "brainstem"}
ANALYTICAL_REGIONS = {"prefrontal_cortex", "acc", "basal_ganglia", "cerebellum", "wernicke_area"}
SOCIAL_REGIONS = {"medial_pfc", "tpj", "insula"}
MEMORY_REGIONS = {"hippocampus"}
LANGUAGE_REGIONS = {"wernicke_area", "broca_area", "spt", "psts"}
SENSORY_REGIONS = {"thalamus", "visual_cortex", "auditory_cortex_left", "auditory_cortex_right", "angular_gyrus"}

# Default gain for all regions
DEFAULT_GAIN = 0.5

# Minimum gain — regions below this use minimal processing (skip LLM, use defaults)
MIN_ACTIVE_GAIN = 0.3


def compute_activation_profile(
    comprehension: dict,
    emotional_tag: dict | None = None,
    has_procedure: bool = False,
) -> dict[str, float]:
    """Compute per-region activation gain [0.0, 1.0] based on input content.

    Returns a dict mapping region_name -> activation_gain.
    Higher gain = region processes fully (LLM calls, detailed analysis).
    Lower gain = region uses minimal processing (defaults, heuristics).

    Args:
        comprehension: Wernicke output with intent, complexity, keywords, etc.
        emotional_tag: Amygdala output with valence, arousal (or None).
        has_procedure: Whether a procedural cache hit was found.
    """
    gains: dict[str, float] = {}

    intent = comprehension.get("intent", "statement")
    complexity = comprehension.get("complexity", "moderate")
    arousal = 0.0
    valence = 0.0
    if emotional_tag:
        arousal = emotional_tag.get("arousal", 0.0) if isinstance(emotional_tag, dict) else getattr(emotional_tag, "arousal", 0.0)
        valence = emotional_tag.get("valence", 0.0) if isinstance(emotional_tag, dict) else getattr(emotional_tag, "valence", 0.0)

    abs_valence = abs(valence)

    # ── Emotional modulation (Pessoa 2008) ──
    # High arousal / strong valence → emotional regions UP, analytical DOWN
    emotional_drive = arousal * 0.6 + abs_valence * 0.4
    for r in EMOTIONAL_REGIONS:
        gains[r] = DEFAULT_GAIN + emotional_drive * 0.5  # 0.5 → 1.0

    # ── Analytical modulation (Duncan 2010) ──
    # Complex tasks, questions, explanations → analytical regions UP
    analytical_drive = 0.0
    if complexity in ("complex", "very_complex"):
        analytical_drive += 0.4
    elif complexity == "moderate":
        analytical_drive += 0.2
    if intent in ("question", "explanation", "analysis", "debug", "request"):
        analytical_drive += 0.3
    for r in ANALYTICAL_REGIONS:
        gains[r] = DEFAULT_GAIN + analytical_drive * 0.8

    # ── Social modulation (Barrett & Satpute 2013) ──
    if intent in ("sharing", "venting", "social", "greeting", "farewell"):
        for r in SOCIAL_REGIONS:
            gains[r] = max(gains.get(r, DEFAULT_GAIN), 0.7)

    # ── Memory modulation ──
    # Questions and recall-type intents boost hippocampus
    if intent in ("question", "recall", "reference"):
        for r in MEMORY_REGIONS:
            gains[r] = max(gains.get(r, DEFAULT_GAIN), 0.8)

    # ── Language regions: always moderate-to-high ──
    for r in LANGUAGE_REGIONS:
        gains[r] = max(gains.get(r, DEFAULT_GAIN), 0.5)

    # ── Sensory: baseline unless multimodal ──
    for r in SENSORY_REGIONS:
        gains.setdefault(r, 0.4)

    # ── Simple/procedural inputs: suppress most regions ──
    if has_procedure and complexity == "simple":
        for r in ANALYTICAL_REGIONS:
            gains[r] = min(gains.get(r, DEFAULT_GAIN), 0.3)
        for r in EMOTIONAL_REGIONS:
            gains[r] = min(gains.get(r, DEFAULT_GAIN), 0.4)

    # ── Inverse: low emotion → suppress emotional regions ──
    if emotional_drive < 0.2:
        for r in EMOTIONAL_REGIONS:
            gains[r] = min(gains.get(r, DEFAULT_GAIN), 0.4)

    # ── Inverse: simple tasks → suppress analytical regions ──
    if complexity == "simple" and intent in ("greeting", "confirmation", "farewell"):
        for r in ANALYTICAL_REGIONS:
            gains[r] = min(gains.get(r, DEFAULT_GAIN), 0.3)

    # Clamp all gains to [0.1, 1.0]
    return {r: max(0.1, min(1.0, g)) for r, g in gains.items()}


def should_full_process(region_name: str, profile: dict[str, float]) -> bool:
    """Check if a region should do full processing (LLM calls etc).

    Regions below MIN_ACTIVE_GAIN should use minimal/default processing.
    """
    return profile.get(region_name, DEFAULT_GAIN) >= MIN_ACTIVE_GAIN
