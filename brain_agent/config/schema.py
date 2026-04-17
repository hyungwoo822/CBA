"""Pydantic configuration schema for brain-agent."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    api_key: str = ""
    api_base: str | None = None
    model: str = ""


class GraphAnalysisConfig(BaseModel):
    max_hub_concepts: int = Field(default=5, ge=1, le=20)
    max_surprising: int = Field(default=3, ge=1, le=10)
    max_communities_shown: int = Field(default=5, ge=1, le=20)


class MemoryConfig(BaseModel):
    working_capacity: int = Field(default=4, ge=2, le=9)
    consolidation_threshold: int = Field(default=3, ge=1)
    homeostatic_factor: float = Field(default=0.97, ge=0.5, le=1.0)
    pruning_threshold: float = Field(default=0.05, ge=0.0, le=0.5)
    embedding_model: str = "all-MiniLM-L6-v2"
    graph_analysis: GraphAnalysisConfig = Field(default_factory=GraphAnalysisConfig)


class RetrievalWeights(BaseModel):
    alpha: float = Field(default=0.25, description="recency")
    beta: float = Field(default=0.30, description="relevance")
    gamma: float = Field(default=0.20, description="importance")
    delta: float = Field(default=0.10, description="frequency")
    epsilon: float = Field(default=0.15, description="context_match")


class DreamingConfig(BaseModel):
    """Memory dreaming (recall-based promotion) configuration."""
    mode: str = Field(default="core", description="off | core | rem | deep")
    check_interval_turns: int | None = Field(default=None, description="Override preset interval")
    min_score: float | None = Field(default=None, description="Override preset min_score")
    min_recall_count: int | None = Field(default=None, description="Override preset min_recall_count")
    min_unique_queries: int | None = Field(default=None, description="Override preset min_unique_queries")


class ExtractionConfig(BaseModel):
    """Knowledge-layer extraction orchestrator configuration."""
    triage_model: str = "auto"
    extract_model: str = "auto"
    temporal_classify_model: str = "auto"
    refine_model: str = "auto"
    max_retry: int = 1
    enable_severity_block: bool = True
    promotion_threshold_n: int = Field(default=3, ge=1, le=50)
    max_open_questions_per_extraction: int = Field(default=3, ge=1)
    fok_similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    pattern_separation_label_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    pattern_separation_window_hours: int = Field(default=24, ge=1)
    # In expression mode, severe extraction findings should not silence the
    # memory-only answer path; demote block to append unless disabled.
    expression_override_block: bool = True


class WorkspaceConfig(BaseModel):
    """Workspace defaults."""
    default_decay_policy: str = "normal"
    vault_size_threshold_mb: int = Field(default=10, ge=1)
    vault_dir: str = "vault"


class AgentConfig(BaseModel):
    model: str = "openai/gpt-4o-mini"
    provider: str = "auto"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_tool_iterations: int = 40


class DashboardConfig(BaseModel):
    enabled: bool = False
    port: int = 3000
    event_buffer_size: int = 1000


class ModalityConfig(BaseModel):
    """Per-modality LLM model configuration.
    Each sensory modality can use a different model optimized for that input type.
    Empty string = fall back to agent.model.
    """
    vision_model: str = ""   # V1 visual cortex (e.g. "openai/gpt-4o")
    stt_model: str = ""      # A1 auditory cortex (e.g. "whisper-1")
    text_model: str = ""     # Wernicke/PFC/Broca (e.g. "openai/gpt-4o-mini")


class MCPServerConfig(BaseModel):
    """Single MCP server connection config."""
    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # http/sse transport
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    transport: str | None = None  # "sse" | "streamable-http" | auto-detect


class MCPConfig(BaseModel):
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    enabled_builtins: list[str] = Field(default_factory=list)


class MiddlewareLayerConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)


class MiddlewareConfig(BaseModel):
    meninges: MiddlewareLayerConfig = Field(default_factory=MiddlewareLayerConfig)
    myelin: MiddlewareLayerConfig = Field(
        default_factory=lambda: MiddlewareLayerConfig(enabled=["myelin_sheath"])
    )
    barrier: MiddlewareLayerConfig = Field(default_factory=MiddlewareLayerConfig)


class TelegramConfig(BaseModel):
    enabled: bool = False
    token: str = ""


class DiscordConfig(BaseModel):
    enabled: bool = False
    token: str = ""


class ChannelsConfig(BaseModel):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)


class TracingConfig(BaseModel):
    """LLM call tracing configuration (LangSmith / LangFuse)."""
    enabled: bool = True
    provider: str = "langfuse"  # "langfuse" | "langsmith"
    project_name: str = "CBA"
    api_key: str = ""


class BrainAgentConfig(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    dreaming: DreamingConfig = Field(default_factory=DreamingConfig)
    retrieval: RetrievalWeights = Field(default_factory=RetrievalWeights)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    modality: ModalityConfig = Field(default_factory=ModalityConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainAgentConfig:
        return cls(**data)
