"""LLM provider inventory for the dashboard model selector."""
from __future__ import annotations

import os
from typing import Any


PROVIDER_ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
    "ollama": [],
}


def _fetch_model_list() -> list[dict[str, Any]]:
    """Return LiteLLM's model inventory, isolated for tests."""
    try:
        import litellm

        return list(getattr(litellm, "model_list", []) or [])
    except Exception:
        return []


def _vendor_available(vendor: str) -> tuple[bool, str | None]:
    keys = PROVIDER_ENV_KEYS.get(vendor, [])
    if not keys:
        return True, None
    if any(os.environ.get(key) for key in keys):
        return True, None
    return False, f"missing env: {'|'.join(keys)}"


def build_inventory(default_model: str) -> dict[str, Any]:
    entries = _fetch_model_list()
    if not entries:
        vendor = default_model.split("/", 1)[0] if "/" in default_model else "openai"
        ok, reason = _vendor_available(vendor)
        item: dict[str, Any] = {
            "id": default_model,
            "vendor": vendor,
            "available": ok,
        }
        if reason:
            item["reason"] = reason
        return {"default_model": default_model, "available": [item]}

    available = []
    seen: set[tuple[str, str]] = set()
    for model in entries:
        if isinstance(model, str):
            model_id = model
            vendor = _vendor_from_model_id(model_id)
        elif isinstance(model, dict):
            vendor = (
                model.get("litellm_provider")
                or model.get("custom_llm_provider")
                or model.get("provider")
                or _vendor_from_model_id(str(model.get("model_name") or model.get("model") or ""))
            )
            model_id = model.get("model_name") or model.get("model") or model.get("id")
        else:
            continue
        if not model_id:
            continue
        key = (str(vendor), str(model_id))
        if key in seen:
            continue
        seen.add(key)
        ok, reason = _vendor_available(str(vendor))
        item = {"id": str(model_id), "vendor": str(vendor), "available": ok}
        if reason:
            item["reason"] = reason
        available.append(item)
    return {"default_model": default_model, "available": available}


def _vendor_from_model_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    lowered = model_id.lower()
    if lowered.startswith("gpt") or lowered.startswith("o1") or lowered.startswith("o3"):
        return "openai"
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "gemini"
    return "unknown"
