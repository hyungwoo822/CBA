"""Reflection -- higher-level insight generation from recent memories.

Brain mapping: During DMN mode, medial PFC generates self-referential
insights from recent experiences.  Park et al. (2023) Generative Agents.
"""
from __future__ import annotations

import re

MIN_EPISODES_FOR_REFLECTION = 5


def build_reflection_prompt(
    recent_episodes: list[dict],
    max_episodes: int = 20,
) -> str:
    """Build a prompt for PFC to generate higher-level insights.

    Parameters
    ----------
    recent_episodes : list[dict]
        Recent episodes with at least a ``content`` key.
    max_episodes : int
        Cap on the number of episodes included in the prompt.

    Returns
    -------
    str
        A prompt string ready to be sent to the PFC callback.
    """
    episodes = recent_episodes[:max_episodes]
    contents = "\n".join(f"- {ep['content']}" for ep in episodes)
    return (
        "Review these recent experiences and generate 1-3 higher-level insights, "
        "patterns, or lessons learned. Focus on what is generally true, not specific events.\n\n"
        f"Recent experiences:\n{contents}\n\n"
        "Respond with one insight per line, each wrapped in <insight>...</insight> tags."
    )


def parse_insights(response: str) -> list[str]:
    """Parse ``<insight>`` tags from a PFC reflection response.

    Returns
    -------
    list[str]
        Extracted insight strings (may be empty if no tags found).
    """
    return [
        m.strip()
        for m in re.findall(r"<insight>(.*?)</insight>", response, re.DOTALL)
        if m.strip()
    ]
