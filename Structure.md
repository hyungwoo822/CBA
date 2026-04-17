# CBA Project Structure

Fast navigation index for this repo. Read this **before** exploring the codebase or writing new code, and update it whenever you add/rename/delete files or change a module's responsibility. Keep each entry to one line. If a subdir grows beyond ~15 files, split it conceptually in the appropriate section.

**Last updated:** 2026-04-17
**Python:** 3.11+ (aiosqlite WAL, pytest-asyncio). **Frontend:** React 19 + Three.js + Vite + Zustand.

---

## Root

| Path | Purpose |
|---|---|
| `brain_agent/` | Main Python package — the agent runtime |
| `dashboard/` | React/Three.js frontend (Vite, TypeScript) |
| `tests/` | Pytest suite, mirrors `brain_agent/` layout |
| `docs/` | **Gitignored.** Design specs, implementation plans, temp notes |
| `data/` | **Gitignored.** Runtime SQLite + ChromaDB + vault files |
| `assets/` | Banner / images referenced by README |
| `graphify/`, `openclaw/` | **Gitignored.** Reference-only external projects |
| `pyproject.toml` | Package config, deps, entry points (`brain-agent` CLI) |
| `uv.lock` | Dependency lockfile |
| `README.md` | Public landing doc |
| `Structure.md` | **This file.** |
| `CLAUDE.md` | Guidance for Claude Code sessions |
| `.env.example` | Env var template (provider keys, feature flags) |
| `.gitignore` | Excludes `docs/`, `data/`, `.env`, reference projects |
| `LICENSE` | MIT |

---

## `tests/memory/` — Memory Test Additions

| File | Purpose |
|---|---|
| `test_personal_adapter.py` | Phase 4 adapter passthrough, node rendering, write-back, and round-trip tests |
| `test_phase4_smoke.py` | Phase 4 regression smoke for legacy `identity_facts` callers |
| `test_phase6_decay_policy.py` | Phase 6 consolidation policy resolution plus transfer/homeostatic decay tests |
| `test_phase6_dream_engine.py` | Phase 6 dreaming origin workspace tracking and all-workspaces parity tests |
| `test_phase6_edge_decay.py` | Phase 6 semantic edge decay/prune tests for workspace, importance, and never_decay |
| `test_phase6_forgetting.py` | Phase 6 ForgettingEngine policy, never_decay, and importance-scaling tests |
| `test_phase6_smoke.py` | Phase 6 mixed-workspace end-to-end decay smoke tests |
| `test_retrieve_with_contradictions.py` | Phase 5 retrieval post-processing tests for contradictions and reconstruction gaps |
| `test_workspace_awareness_backward_compat.py` | Phase 5 signature compatibility tests for workspace-aware exports and staging |

## `tests/pipeline/` — Phase 5 Pipeline Integration Tests

| File | Purpose |
|---|---|
| `__init__.py` | Package marker for pipeline integration tests |
| `test_pipeline_extraction_integration.py` | Orchestrator wiring, response-mode branching, workspace, and S1/S2 propagation tests |
| `test_pipeline_expression_override.py` | Expression-mode `block` to `append` override coverage |
| `test_phase5_smoke.py` | End-to-end Phase 5 normal/block/append smoke tests |
| `test_pipeline_dashboard_events.py` | Clarification-request event emission test for block mode |

## `tests/config/` — Config Test Additions

| File | Purpose |
|---|---|
| `test_extraction_config.py` | ExtractionConfig and WorkspaceConfig defaults + BrainAgentConfig wiring |

## `tests/regions/` — Phase 5 Region Test Additions

| File | Purpose |
|---|---|
| `test_broca_block_mode.py` | Broca block-mode clarification question formatting |
| `test_wernicke_workspace_hint.py` | Wernicke pragmatic workspace hint parsing and M1 annotation |
| `test_expression_mode_instruction.py` | PFC expression-mode workspace/contradiction/gap wording |

---

## `brain_agent/` — Package Entry Points

| File | Purpose |
|---|---|
| `__init__.py` | Public exports (`BrainAgent`) |
| `__main__.py` | `python -m brain_agent` entry |
| `agent.py` | **Main class `BrainAgent`** — orchestrates pipeline, memory, channels, lifespan |
| `pipeline.py` | **`ProcessingPipeline`** — 7-phase neural pipeline with Phase 5 extraction orchestrator integration |

## `brain_agent/regions/` — 23 Brain Regions

Each region is a self-contained class in its own file. Regions mix LLM-backed and algorithmic computation.

**LLM-backed (6):** `prefrontal.py`, `broca.py`, `wernicke.py`, `amygdala.py`, `visual_cortex.py`, `auditory_cortex.py`
**Algorithmic (17):** `acc.py`, `angular_gyrus.py`, `basal_ganglia.py`, `brainstem.py`, `cerebellum.py`, `corpus_callosum.py`, `goal_tree.py`, `hypothalamus.py`, `insula.py`, `motor_cortex.py`, `mpfc.py`, `psts.py`, `salience_network.py`, `spt.py`, `thalamus.py`, `tpj.py`, `vta.py`
**Shared:** `base.py` — base Region class, injection interfaces, neuromodulator access

## `brain_agent/memory/` — Memory Subsystem

**Six-layer CLS (existing):**

| File | Purpose |
|---|---|
| `sensory_buffer.py` | Per-cycle transient buffer (Sperling) |
| `working_memory.py` | 4-slot Baddeley workspace |
| `hippocampal_staging.py` | SQLite staging, ACh-modulated encoding |
| `episodic_store.py` | SQLite episodes with Ebbinghaus decay |
| `semantic_store.py` | ChromaDB + SQLite knowledge graph (hybrid) |
| `procedural_store.py` | SQLite DA-gated skill patterns |
| `brain_state.py` | SQLite neuromodulator + region activation persistence |

**Engines:**

| File | Purpose |
|---|---|
| `consolidation.py` | `ConsolidationEngine` — staging→semantic/episodic promotion |
| `forgetting.py` | `ForgettingEngine` — interference + decay |
| `retrieval.py` | `RetrievalEngine` — semantic + episodic search |
| `dreaming.py` | `DreamingEngine` + `RecallTracker` — cross-domain association |
| `manager.py` | **`MemoryManager`** — facade; owns all stores + engines + personal adapter |

**Knowledge graph analysis:**

| File | Purpose |
|---|---|
| `graph_analysis.py` | Leiden clustering, hubs, bridges, cell assemblies, diffs |
| `narrative_consolidation.py` | Story-graph promotion during consolidation |
| `semantic_extractor.py` | Lightweight entity/relation extraction (pre-PSC) |
| `embedding_cache.py` | SHA256-keyed LRU embedding cache |
| `reflection.py` | Stub — future reflection mechanism |

**Knowledge Layer:**

| File | Purpose |
|---|---|
| `workspace_store.py` | Multi-workspace registry + session binding |
| `ontology_store.py` | Node/relation type registry with 4-tier confidence + proposal queue |
| `ontology_seed.py` | Universal ontology constants (7 node + 10 relation types) |
| `personal_adapter.py` | Personal workspace adapter bridging `identity_facts` to Person workspace nodes |
| `raw_vault.py` | SHA256-addressed raw input vault for lossless source storage |
| `contradictions_store.py` | Workspace-scoped contradiction queue and subject batch lookup |
| `open_questions_store.py` | Workspace-scoped clarifying question queue |

**Planned (see `docs/knowledge_layer_plan.md`):**

| Planned file | Phase | Purpose |
|---|---|---|
| `templates/software_project.py` etc. | 7 | Domain ontology templates |

## `brain_agent/extraction/` — Multi-stage Extractor

| File | Purpose |
|---|---|
| `config.py` | Dataclass config for extractor stages and response-mode behavior |
| `types.py` | Dataclass contracts for triage, temporal resolution, validation, severity, and extraction results |
| `triage.py` | Stage 1 input classification, skip-stage logic, and workspace hint handling |
| `extractor.py` | Stage 2 ontology-aware structured extraction from user text |
| `temporal_resolver.py` | Stage 2.5 temporal update/contradiction classification |
| `validator.py` | Stage 3 contradiction, missing-property, pattern-separation, and FOK validation |
| `severity.py` | Stage 4 response-mode decision from contradictions and open questions |
| `refiner.py` | Stage 5 personal-workspace response polishing |
| `orchestrator.py` | Multi-stage extraction coordinator with raw vault, staging, contradictions, questions, and proposals |
| `_mock_llm.py` | Recording LLM provider used by extraction tests |
| `__init__.py` | Extraction package marker |

## `brain_agent/migrations/` — Schema Migration Runner

| File | Purpose |
|---|---|
| `__init__.py` | Exports `MigrationRunner` |
| `runner.py` | Discovers + applies migration modules in order, records in `schema_version` |
| `steps/m000_init_schema_version.py` | Bootstrap — creates `schema_version` table in `brain_state.db` |
| `steps/m001_workspace_columns.py` (planned) | Phase 1 ALTER TABLEs |

## `brain_agent/core/` — Shared Primitives

| File | Purpose |
|---|---|
| `signals.py` | Signal + NeuralSignal dataclasses |
| `session.py` | `SessionManager` — interaction lifecycle |
| `router.py` | `ThalamicRouter` — network-mode-aware region routing |
| `network_modes.py` | DMN/ECN/SN mode detection |
| `neuromodulators.py` | 7-NT state (DA, NE, 5-HT, ACh, CORT, EPI, GABA) |
| `neuromodulator_controller.py` | NT updates per event |
| `predictor.py` | Forward-model prediction engine |
| `workspace.py` | `GlobalWorkspace` (Baars) — broadcast gating (distinct from knowledge-layer workspace) |
| `embeddings.py` | Embedding fn factory (provider-backed) |
| `activation_profile.py` | Region activation time-series |
| `temporal.py` | Time utilities |

## `brain_agent/providers/` — LLM Abstraction (Vendor-Agnostic)

| File | Purpose |
|---|---|
| `base.py` | `LLMProvider` ABC + `LLMResponse` / `ToolCallRequest` dataclasses |
| `litellm_provider.py` | LiteLLM-backed provider (Anthropic/OpenAI/Gemini/Grok/Ollama/…) |
| `myelinated.py` | Caching decorator over LLMProvider |

**Rule:** every LLM call goes through `LLMProvider.chat(...)`. No vendor SDK imports (`anthropic`, `openai`) outside this directory.

## `brain_agent/dashboard/` — Backend Dashboard

| File | Purpose |
|---|---|
| `server.py` | FastAPI app + `/ws` WebSocket + REST endpoints (`/api/*`) |
| `emitter.py` | Event emitter (region_activation, memory_event, knowledge_update, clarification_requested; Phase 8 adds workspace/curation events) |

## `brain_agent/cli/` — CLI

| File | Purpose |
|---|---|
| `commands.py` | `brain-agent run / dashboard / memory stats / …` — Phase 0 plan adds `workspace/ontology/questions/contradictions` subcommands |

## `brain_agent/config/` — Configuration

| File | Purpose |
|---|---|
| `schema.py` | Pydantic config models including extraction and workspace settings (loaded from env + `.env`) |

## `brain_agent/channels/` — External Chat Adapters

| File | Purpose |
|---|---|
| `base.py` | `Channel` ABC |
| `manager.py` | `ChannelManager` — connects/broadcasts |
| `discord_adapter.py` | Discord bot |
| `telegram_adapter.py` | Telegram bot |

## `brain_agent/mcp/` — Model Context Protocol

| File | Purpose |
|---|---|
| `client.py` | MCP client |
| `knowledge_server.py` | MCP server exposing `query_graph`, `get_neighbors`, etc. |
| `proxy.py`, `registry.py`, `transport.py` | Proxy + tool registry + transport layer |

## `brain_agent/tools/` — Tool Registry (Generic Tool Calls)

| File | Purpose |
|---|---|
| `base.py` | `Tool` ABC |
| `registry.py` | Tool registry |
| `builtin/file_read.py`, `file_write.py`, `shell.py`, `web_fetch.py`, `web_search.py` | Built-in tools |

## `brain_agent/middleware/` — Request/Response Middleware

| Dir/File | Purpose |
|---|---|
| `base.py` / `registry.py` | Middleware ABC + registry |
| `barrier/blood_brain_barrier.py`, `microglial_defense.py`, `synaptic_timeout.py` | Input sanitization + timeouts |
| `meninges/arachnoid_tracer.py`, `dura_mater.py` | Wrapping / tracing |
| `myelin/sheath.py` | Cache layer (pairs with `providers/myelinated.py`) |

## `brain_agent/tracing/` — Observability

| File | Purpose |
|---|---|
| `manager.py` | Tracing orchestration |
| `langsmith_tracer.py` | LangSmith backend |
| `langfuse_tracer.py` | LangFuse backend |

---

## `dashboard/` — React Frontend

| Path | Purpose |
|---|---|
| `package.json` / `vite.config.ts` / `tsconfig*.json` | Build config |
| `index.html` | Vite entry |
| `src/main.tsx` / `App.tsx` / `App.css` / `index.css` | App root |
| `src/stores/brainState.ts` | Zustand store (regions, neuromodulators, events, chat, workspace — Phase 8 extends) |
| `src/hooks/useWebSocket.ts` | WS subscription |
| `src/hooks/useDraggable.ts` | Modal dragging |
| `src/constants/brainRegions.ts` | 3D coords + region metadata |
| `src/utils/fresnelMaterial.ts` | Three.js shader |

### `dashboard/src/components/`

| Component | Purpose |
|---|---|
| `BrainScene.tsx` / `BrainModel.tsx` / `RegionBubble.tsx` | 3D brain scene + region spheres |
| `CurvedConnections.tsx` / `SignalParticles.tsx` | Neural connections + flowing particles |
| `HUD.tsx` | Top overlay — network mode + NT bars |
| `MemoryPanel.tsx` / `MemoryFlowBar.tsx` | Memory layer counts + flow |
| `KnowledgeGraphPanel.tsx` / `KnowledgeGraphModal.tsx` | Force-directed KG viz (Phase 8 adds workspace filter) |
| `EventLog.tsx` | WS event stream |
| `ChatInput.tsx` / `BrainResponseBubble.tsx` | Conversation UI |
| `InteractionModeToggle.tsx` | question/expression mode switch |
| `ChannelToggle.tsx` | Discord/Telegram toggle |
| `AudioOrb.tsx` | Voice input indicator |
| `RegionDetailPanel.tsx` / `ProfileEditModal.tsx` | Region detail + identity edit modals |

Planned additions (Phase 8): `WorkspaceSelector.tsx`, `CurationInbox.tsx`, `QuestionCard.tsx`, `ContradictionCard.tsx`, `ProposalCard.tsx`, `RawVaultPanel.tsx`, `TimelineView.tsx`, `ExportPreviewModal.tsx`, `ModelSelector.tsx`.

---


## Entry Points

| Use case | Start here |
|---|---|
| Run the agent programmatically | `brain_agent/agent.py::BrainAgent` |
| Understand the request pipeline | `brain_agent/pipeline.py::ProcessingPipeline.process_request` |
| Add a new LLM provider / model | `brain_agent/providers/base.py` + litellm model string |
| Add a brain region | `brain_agent/regions/base.py` → new file in `regions/` → register in `pipeline.py` |
| Add a memory store | `brain_agent/memory/manager.py::MemoryManager.__init__` + own `.py` file + migration in `migrations/steps/` |
| Add a REST endpoint | `brain_agent/dashboard/server.py` |
| Add a WS event | `brain_agent/dashboard/emitter.py` + consume in `dashboard/src/stores/brainState.ts` |
| Add a CLI command | `brain_agent/cli/commands.py` |
| Implement Phase N of knowledge layer | `docs/superpowers/plans/2026-04-17-phase-N-*.md` |

---

## Update Discipline

When you finish a code change, update this file if any of the following is true:
1. A new file or directory was added
2. A file was renamed, moved, or deleted
3. A module's one-line purpose changed
4. A new planned file became real (move it out of "Planned" sub-table)

Do not re-explore the tree to verify — trust the diff you just made. Append to the relevant section's table, keep the one-line-per-file rule, and bump `Last updated:` at the top.
