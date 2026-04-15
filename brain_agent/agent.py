"""BrainAgent -- the public API for the brain-agent framework."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from brain_agent.config.schema import BrainAgentConfig
from brain_agent.core.embeddings import EmbeddingService
from brain_agent.core.session import SessionManager
from brain_agent.memory.manager import MemoryManager
from brain_agent.middleware import MiddlewareRegistry
from brain_agent.middleware.base import MiddlewareContext
from brain_agent.mcp.registry import MCPRegistry
from brain_agent.pipeline import ProcessingPipeline, PipelineResult
from brain_agent.providers.base import LLMProvider
from brain_agent.providers.litellm_provider import LiteLLMProvider
from brain_agent.providers.myelinated import MyelinatedProvider
from brain_agent.tools.registry import ToolRegistry
from brain_agent.tracing import TracingManager

logger = logging.getLogger(__name__)


class BrainAgent:
    """Main entry point for the brain-agent framework.

    Usage:
        agent = BrainAgent(provider="openai", model="gpt-4o-mini")
        response = await agent.process("Find the bug in auth module")
    """

    def __init__(
        self,
        provider: str = "auto",
        model: str | None = None,
        api_key: str | None = None,
        config: BrainAgentConfig | None = None,
        data_dir: str | None = None,
        use_mock_embeddings: bool = False,
        region_overrides: dict | None = None,
    ):
        self.config = config or BrainAgentConfig()
        if model:
            self.config.agent.model = model
        if provider != "auto":
            self.config.agent.provider = provider
        if api_key:
            self.config.provider.api_key = api_key

        self._data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(self._data_dir, exist_ok=True)

        self._embedding_service = EmbeddingService(
            model_name=self.config.memory.embedding_model,
            use_mock=use_mock_embeddings,
        )
        self._embed_fn = self._embedding_service.embed

        # Memory system — single MemoryManager shared with pipeline
        self.memory = MemoryManager(
            db_dir=self._data_dir,
            embed_fn=self._embed_fn,
            working_capacity=self.config.memory.working_capacity,
            consolidation_threshold=self.config.memory.consolidation_threshold,
        )
        self.session_manager = SessionManager(
            db_path=os.path.join(self._data_dir, "sessions.db"),
            embed_fn=self._embed_fn,
        )

        # LLM provider — supports explicit api_key OR auto-detection
        # from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY,
        # GEMINI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY).
        explicit_key = self.config.provider.api_key or None
        env_key_available = explicit_key is not None or any(
            os.environ.get(k)
            for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                       "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY")
        )
        if env_key_available:
            self._llm_provider = LiteLLMProvider(
                model=self.config.agent.model,
                api_key=explicit_key,  # None = let litellm auto-detect from env
            )
        else:
            self._llm_provider = None

        # Tools — load built-ins from config
        self.tools = ToolRegistry()
        self.tools.load_builtins(self.config.tools.enabled_builtins)

        # MCP — will be initialized async in initialize()
        self._mcp_registry = MCPRegistry()

        # ── Neural sheath system (3-layer middleware) ──
        # Anatomical mapping:
        #   meninges — pipeline-level protective membranes (DuraMater, ArachnoidTracer)
        #   myelin   — LLM-level signal insulation (MyelinSheath)
        #   barrier  — tool-level selective permeability (BBB, SynapticTimeout, Microglial)
        mw_registry = MiddlewareRegistry()
        self._meninges_mw = mw_registry.build_chain(self.config.middleware.meninges.enabled)
        self._myelin_mw = mw_registry.build_chain(self.config.middleware.myelin.enabled)
        self._barrier_mw = mw_registry.build_chain(self.config.middleware.barrier.enabled)

        # ── Tracing (LangSmith observability) ──
        self.tracing = TracingManager(self.config.tracing)

        # Myelinate the LLM provider — wraps every chat() call with
        # myelin middleware (token counting, etc.) transparently.
        # No brain region needs to know about the sheath.
        if self._llm_provider and self.config.middleware.myelin.enabled:
            self._llm_provider = MyelinatedProvider(
                inner=self._llm_provider,
                myelin=self._myelin_mw,
            )

        # Pipeline — receives MemoryManager so memory is integrated into
        # every processing stage, not bolted on afterward.
        # Also receives the barrier middleware chain for tool-level gating.
        self.pipeline = ProcessingPipeline(
            memory=self.memory,
            llm_provider=self._llm_provider,
            barrier_mw=self._barrier_mw,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all subsystems. Must be called before process()."""
        await self.memory.initialize()
        # Restore brain state from previous session (McEwen 2007: allostatic load persists)
        await self.pipeline.restore_brain_state()
        await self.session_manager.initialize()
        await self.session_manager.start_session()

        # Connect MCP servers and bridge their tools into ToolRegistry
        if self.config.mcp.servers:
            await self._mcp_registry.initialize(self.config.mcp.servers)
            self._mcp_registry.bridge_to_tool_registry(self.tools)

        self._initialized = True

    async def close(self) -> None:
        """Shutdown and cleanup."""
        # Session end triggers consolidation (= brain "sleep")
        if await self.memory.consolidation.should_consolidate():
            await self.memory.consolidate()
        await self._mcp_registry.shutdown()
        await self.session_manager.close()
        await self.memory.close()
        self._initialized = False

    async def process(self, text: str, image: bytes | None = None, audio: bytes | None = None) -> PipelineResult:
        """Process a user request through the full brain pipeline.

        The meninges middleware wraps this entire cycle — like the dura
        mater enveloping the brain, it monitors input/output and timing
        at the outermost layer.
        """
        if not self._initialized:
            await self.initialize()

        # Check session boundary
        if self.session_manager.should_start_new_session(text):
            await self.session_manager.close_session()
            # Consolidate on session end (= "sleep" replay)
            if await self.memory.consolidation.should_consolidate():
                await self.memory.consolidate()
            await self.session_manager.start_session()

        # Tick interaction
        interaction_id = await self.session_manager.on_interaction(text)
        session_id = (
            self.session_manager.current_session.id
            if self.session_manager.current_session
            else ""
        )
        self.memory.set_context(interaction_id, session_id)

        # ── Determine modality ──
        modality = "text"
        if image:
            modality = "image"
        elif audio:
            modality = "audio"

        # ── Trace: start root run ──
        trace_run = self.tracing.start_request_trace(
            text=text, session_id=session_id,
            interaction_id=interaction_id, modality=modality,
        )

        # ── Meninges wrap: pipeline-level protective membrane ──
        context = MiddlewareContext(data={
            "user_input": text,
            "image": image,
            "audio": audio,
        })

        async def _pipeline_core(ctx: MiddlewareContext) -> MiddlewareContext:
            result = await self.pipeline.process_request(
                ctx["user_input"],
                image=ctx.get("image"),
                audio=ctx.get("audio"),
                trace_run=trace_run,
            )
            ctx["result"] = result
            return ctx

        context = await self._meninges_mw.execute(context, _pipeline_core)
        result = context["result"]

        # ── Trace: end root run ──
        self.tracing.end_request_trace(trace_run, {
            "response": result.response,
            "network_mode": result.network_mode,
            "signals_processed": result.signals_processed,
        })

        return result

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # Convenience session context manager
    class _SessionCtx:
        def __init__(self, agent: BrainAgent):
            self._agent = agent

        async def __aenter__(self):
            if not self._agent._initialized:
                await self._agent.initialize()
            return self

        async def __aexit__(self, *args):
            await self._agent.close()

        async def send(self, text: str) -> PipelineResult:
            return await self._agent.process(text)

    def session(self) -> _SessionCtx:
        """Context manager for a conversation session."""
        return self._SessionCtx(self)
