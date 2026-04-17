# AGENTS.md

Guidance for codex sessions in this repo.

## Structure.md — read before code work, update after

**Before writing or modifying any code in this project:** read `Structure.md` at the repo root. It is a hand-maintained one-line-per-file index of the whole codebase. Use it to locate the right file, check whether a helper already exists, or confirm where a new module belongs — *without* re-exploring the tree. This saves tokens and prevents duplicate scaffolding.

**After a code change is complete** (new file added, file renamed/moved/deleted, or a module's one-line purpose changed): update `Structure.md` to reflect the new state. Append to the relevant section's table, keep the one-line-per-file rule, bump the `Last updated:` date at the top. Do not re-scan the tree to verify — trust your diff.

If `Structure.md` seems out of date or misses a file you just touched, fix it in the same change that touched that file. Don't leave the index drifting.

## Project-specific rules

- **Local-only docs:** `docs/` is gitignored. Plans, specs, and design notes under `docs/superpowers/plans/` and `docs/superpowers/specs/` live on disk only — they are never pushed. Treat them as authoritative for intent, but don't assume they're public.
- **LLM provider abstraction:** Every model call must route through `brain_agent/providers/base.py::LLMProvider`. No vendor SDK imports outside that directory. Config fields accept litellm-style model identifiers (e.g. `"anthropic/claude-sonnet-4-6"`, `"openai/gpt-4o-mini"`, `"ollama/llama3"`); default `"auto"` falls back to `LLMProvider.get_default_model()`.
- **Knowledge layer rollout:** Phased work tracked in `docs/knowledge_layer_plan.md` (spec) + `docs/superpowers/plans/2026-04-17-phase-{0..8}-*.md` (9 TDD plans). Phase 0 foundation is landing. Don't start a later phase's task before its prior phases are complete.
- **Never commit or push without explicit user instruction.** This applies to both main and any feature branch.
