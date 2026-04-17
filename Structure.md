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
| `test_phase9_audit_fixes.py` | Phase 9 audit regression tests for staging defaults, curation confidence labels, and template composition |
| `test_retrieve_with_contradictions.py` | Phase 5 retrieval post-processing tests for contradictions and reconstruction gaps |
| `test_apply_template.py` | Phase 7 template application, composition, canonical confidence, and workspace isolation tests |
| `test_templates_smoke.py` | Phase 7 end-to-end smoke for all bundled templates and universal overlay |
| `test_upgrade_template.py` | Phase 7 template diff, dry-run, upgrade, soft-delete, and downgrade refusal tests |
| `test_workspace_awareness_backward_compat.py` | Phase 5 signature compatibility tests for workspace-aware exports and staging |

## `tests/memory/templates/` — Phase 7 Template Tests

| File | Purpose |
|---|---|
| `__init__.py` | Package marker for template tests |
| `test_template_contents.py` | Pure-data assertions for bundled template dicts and loader metadata |
| `fixtures/__init__.py` | Package marker for upgrade fixture templates |
| `fixtures/software_project_v1_1.py` | Fixture software-project v1.1 minor bump for upgrade tests |
| `fixtures/software_project_v2_0.py` | Fixture software-project v2.0 major bump for upgrade tests |

## `tests/migrations/` — Migration Test Additions

| File | Purpose |
|---|---|
| `test_m002_template_upgrade_columns.py` | Phase 7 migration tests for ontology `deprecated` soft-delete columns |

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
| `test_extraction_config_models.py` | Phase 8 per-stage extraction model defaults and override tests |

## `tests/extraction/` — Extraction Pipeline Tests

| File | Purpose |
|---|---|
| `test_orchestrator_emits.py` | Phase 8 extraction orchestrator curation WebSocket emit hook test |

## `tests/dashboard/` — Dashboard API and UI Integration Tests

| File | Purpose |
|---|---|
| `test_server.py` | Dashboard app and EventBus baseline tests |
| `test_emitter.py` | DashboardEmitter baseline signal-flow test |
| `test_workspace_api.py` | Phase 8 workspace CRUD, current binding, stats, and delete safeguards |
| `test_kg_workspace_filter.py` | Phase 8 knowledge-graph workspace filter and cross-reference edge tests |
| `test_curation_ws_emit.py` | Phase 8 curation WebSocket event emitter tests |
| `test_ontology_api.py` | Phase 8 ontology type/proposal curation API tests |
| `test_curation_api.py` | Phase 8 question and contradiction curation API tests |
| `test_source_api.py` | Phase 8 raw-vault source metadata/raw/text API tests |
| `test_timeline_api.py` | Phase 8 temporal supersede-chain timeline API test |
| `test_export_preview.py` | Phase 8 export preview shape and filter-matrix tests |
| `test_llm_provider_listing.py` | Phase 8 LLM provider inventory endpoint tests |
| `test_phase8_smoke.py` | Phase 8 end-to-end dashboard API smoke test |

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
| `workspace_store.py` | Multi-workspace registry, session binding, and optional workspace_changed emit hook |
| `ontology_store.py` | Node/relation type registry with 4-tier confidence + proposal queue |
| `ontology_seed.py` | Universal ontology constants (7 node + 10 relation types) |
| `personal_adapter.py` | Personal workspace adapter bridging `identity_facts` to Person workspace nodes |
| `raw_vault.py` | SHA256-addressed raw input vault for lossless source storage |
| `contradictions_store.py` | Workspace-scoped contradiction queue and subject batch lookup |
| `open_questions_store.py` | Workspace-scoped clarifying question queue |

**Templates:**

| File | Purpose |
|---|---|
| `templates/__init__.py` | Bundled domain template loader (`get_template`, `list_templates`) |
| `templates/software_project.py` | Software-project ontology template (10 node + 10 relation types) |
| `templates/research_notes.py` | Research-notes ontology template (6 node + 6 relation types) |
| `templates/personal_knowledge.py` | Personal-knowledge ontology template (5 node + 5 relation types) |

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
| `orchestrator.py` | Multi-stage extraction coordinator with raw vault, staging, contradictions, questions, proposals, and curation event emits |
| `_mock_llm.py` | Recording LLM provider used by extraction tests |
| `__init__.py` | Extraction package marker |

## `brain_agent/migrations/` — Schema Migration Runner

| File | Purpose |
|---|---|
| `__init__.py` | Exports `MigrationRunner` |
| `runner.py` | Discovers + applies migration modules in order, records in `schema_version` |
| `steps/m000_init_schema_version.py` | Bootstrap — creates `schema_version` table in `brain_state.db` |
| `steps/m001_workspace_columns.py` | Phase 1 ALTER TABLEs for workspace/provenance columns |
| `steps/m002_template_upgrade_columns.py` | Phase 7 ALTER TABLEs adding ontology type `deprecated` flags |

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
| `providers_inventory.py` | Phase 8 LiteLLM model inventory and env availability mapping for dashboard model selector |

## `brain_agent/dashboard/routers/` — Phase 8 Dashboard Routers

| File | Purpose |
|---|---|
| `__init__.py` | Dashboard router package marker |
| `workspaces.py` | Workspace CRUD, current session binding, stats, and workspace_changed event endpoints |
| `kg.py` | Workspace-aware knowledge-graph visualization endpoint with optional cross-reference edges |
| `ontology.py` | Ontology type listing and proposal approve/reject curation endpoints |
| `curation.py` | Open-question and contradiction list/answer/resolve/dismiss endpoints |
| `sources.py` | Raw-vault source metadata, raw bytes, and extracted text endpoints |
| `timeline.py` | Temporal supersede-chain timeline endpoint |
| `export.py` | MCP-compatible export preview endpoint with confidence, importance, decay, and raw-vault filters |
| `llm.py` | LLM provider inventory endpoint for the dashboard model selector |

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
| `package.json` / `vite.config.ts` / `vitest.config.ts` / `tsconfig*.json` | Build and test config |
| `index.html` | Vite entry |
| `src/main.tsx` / `App.tsx` / `App.css` / `index.css` | App root |
| `src/stores/brainState.ts` | Zustand store (regions, neuromodulators, events, chat, workspace/curation/model UI state) |
| `src/hooks/useWebSocket.ts` | WS subscription including Phase 8 workspace and curation event handlers |
| `src/hooks/useWorkspace.ts` | Phase 8 workspace list/current loading and switching hook |
| `src/hooks/useCurationInbox.ts` | Phase 8 curation inbox data/actions hook for questions, contradictions, and proposals |
| `src/test/setup.ts` | Vitest + Testing Library setup |
| `src/__smoke__.test.ts` | Frontend test-runner smoke test |
| `src/hooks/useDraggable.ts` | Modal dragging |
| `src/hooks/useModalDrag.ts` | Phase 9 wrapper around useDraggable returning style + onMouseDown for kl-modal frames |
| `src/constants/brainRegions.ts` | 3D coords + region metadata |
| `src/utils/fresnelMaterial.ts` | Three.js shader |

### `dashboard/src/components/`

| Component | Purpose |
|---|---|
| `BrainScene.tsx` / `BrainModel.tsx` / `RegionBubble.tsx` | 3D brain scene + region spheres |
| `CurvedConnections.tsx` / `SignalParticles.tsx` | Neural connections + flowing particles |
| `HUD.tsx` | Top overlay for network mode, NT bars, and region activity |
| `MemoryPanel.tsx` / `MemoryFlowBar.tsx` | Memory layer counts + flow |
| `KnowledgeGraphPanel.tsx` / `KnowledgeGraphModal.tsx` | Force-directed KG viz with workspace filter, cross-refs, overlays, and raw-vault hover preview |
| `EventLog.tsx` | WS event stream |
| `ChatInput.tsx` / `BrainResponseBubble.tsx` | Conversation UI |
| `InteractionModeToggle.tsx` | question/expression mode switch |
| `ChannelToggle.tsx` | Discord/Telegram toggle |
| `AudioOrb.tsx` | Voice input indicator |
| `RegionDetailPanel.tsx` / `ProfileEditModal.tsx` | Region detail + identity edit modals |
| `WorkspaceSelector.tsx` | Top-nav workspace dropdown and current workspace switcher |
| `CurationInbox.tsx` | Phase 8 three-tab curation drawer for questions, contradictions, and proposals |
| `QuestionCard.tsx` | Phase 8 open-question answer card |
| `ContradictionCard.tsx` | Phase 8 contradiction resolution card |
| `ProposalCard.tsx` | Phase 8 ontology proposal approve/reject card |
| `RawVaultPanel.tsx` | Phase 8 raw-vault source metadata/text/image preview drawer |
| `TimelineView.tsx` | Phase 8 temporal supersede-chain visualization component |
| `ExportPreviewModal.tsx` | Phase 8 export JSON preview modal with filters, copy, download, and lazy Monaco viewer |
| `ModelSelector.tsx` | Phase 8 per-stage extraction model selector drawer driven by `/api/llm/providers` |
| `TopBarChips.tsx` | Phase 9 single-row chip for workspace/inbox/export/models mounted in top-nav |

### `dashboard/src/components/__tests__/` — Phase 8 Component Tests

| File | Purpose |
|---|---|
| `WorkspaceSelector.test.tsx` | Workspace selector current display and PUT switch action tests |
| `CurationInbox.test.tsx` | Curation inbox tab/count and question-answer action tests |
| `DualChannelSync.test.tsx` | Inbox optimistic removal and WS-style question_answered sync tests |
| `ExportPreviewModal.test.tsx` | Export preview fetch, filter refetch, and clipboard tests |
| `ModelSelector.test.tsx` | Model selector API-driven dropdown and unavailable reason tests |
| `TopBarChips.test.tsx` | Phase 9 TopBarChips placement, inbox-count badge, and open-toggle action tests |

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
