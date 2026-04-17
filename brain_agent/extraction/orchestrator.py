"""ExtractionOrchestrator sequences all Phase 3 extraction stages."""
from __future__ import annotations

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.extractor import Extractor, _ExtractOutput
from brain_agent.extraction.refiner import Refiner
from brain_agent.extraction.severity import compute_response_mode
from brain_agent.extraction.temporal_resolver import TemporalResolver
from brain_agent.extraction.triage import Triage
from brain_agent.extraction.types import (
    ExtractionResult,
    TemporalResolveResult,
    TriageResult,
    ValidationResult,
)
from brain_agent.extraction.validator import Validator
from brain_agent.providers.base import LLMProvider


class ExtractionOrchestrator:
    def __init__(
        self,
        memory,
        llm_provider: LLMProvider,
        config: ExtractionConfig | None = None,
    ):
        self._memory = memory
        self._llm = llm_provider
        self._cfg = config or ExtractionConfig()

        self._triage = Triage(memory.workspace, llm_provider, self._cfg)
        self._extractor = Extractor(memory.ontology, llm_provider, self._cfg)
        self._temporal = TemporalResolver(memory.semantic, llm_provider, self._cfg)
        self._validator = Validator(memory.semantic, memory.ontology, self._cfg)
        self._refiner = Refiner(llm_provider, self._cfg)

    async def extract(
        self,
        text: str,
        session_id: str = "",
        image: bytes | None = None,
        audio: bytes | None = None,
        agent_response: str = "",
        language: str = "auto",
        comprehension: dict | None = None,
    ) -> ExtractionResult:
        triage = await self._triage.classify(text, session_id, comprehension)

        source = await self._memory.raw_vault.ingest(
            workspace_id=triage.target_workspace_id,
            kind="user_utterance",
            data=text.encode("utf-8"),
            mime="text/plain",
            extracted_text=text,
        )

        if 2 in triage.skip_stages:
            extracted = _ExtractOutput(narrative_chunk=text)
        else:
            extracted = await self._extractor.extract(
                text=text,
                workspace_id=triage.target_workspace_id,
            )

        if 2 in triage.skip_stages or not extracted.edges:
            temporal = TemporalResolveResult(new_edges=list(extracted.edges))
        else:
            temporal = await self._temporal.resolve(
                edges=extracted.edges,
                narrative_chunk=extracted.narrative_chunk,
                workspace_id=triage.target_workspace_id,
            )

        if 3 in triage.skip_stages:
            validated = ValidationResult()
        else:
            validated = await self._validator.validate(
                nodes=extracted.nodes,
                edges=temporal.new_edges,
                workspace_id=triage.target_workspace_id,
                narrative_chunk=extracted.narrative_chunk,
                input_kinds=triage.input_kinds,
            )

        severity = compute_response_mode(
            contradictions=validated.contradictions,
            open_questions=validated.open_questions,
            config=self._cfg,
        )

        response_text = ""
        if severity.response_mode != "block":
            workspace = await self._memory.workspace.get_workspace(
                triage.target_workspace_id
            )
            response_text = await self._refiner.refine(
                agent_response=agent_response,
                language=language,
                workspace=workspace,
            )

        await self._persist(
            triage=triage,
            extracted=extracted,
            validated=validated,
            temporal=temporal,
            source=source,
            session_id=session_id,
        )

        clarification_questions = list(severity.clarification_questions)
        if triage.workspace_ask:
            clarification_questions.insert(0, triage.workspace_ask)

        return ExtractionResult(
            workspace_id=triage.target_workspace_id,
            source_id=source["id"],
            narrative_chunk=extracted.narrative_chunk,
            nodes=list(extracted.nodes),
            edges=list(temporal.new_edges),
            contradictions=validated.contradictions,
            open_questions=validated.open_questions,
            new_type_proposals=list(extracted.new_type_proposals),
            response_text=response_text,
            response_mode=severity.response_mode,
            clarification_questions=clarification_questions,
        )

    async def _persist(
        self,
        triage: TriageResult,
        extracted: _ExtractOutput,
        validated: ValidationResult,
        temporal: TemporalResolveResult,
        source: dict,
        session_id: str,
    ) -> None:
        """Persist only to staging and ancillary stores, never semantic/episodic inserts."""
        workspace_id = triage.target_workspace_id
        interaction_id = int(getattr(self._memory, "_interaction_counter", 0) or 0)

        for existing_edge in temporal.reinforced_edges:
            if existing_edge.get("type_id"):
                await self._memory.ontology.increment_occurrence(existing_edge["type_id"])
            if hasattr(self._memory.staging, "reinforce"):
                await _call_with_fallback(
                    self._memory.staging.reinforce,
                    rich_kwargs={
                        "workspace_id": workspace_id,
                        "edge_id": existing_edge.get("id"),
                        "source_id": source["id"],
                    },
                    fallback_args=(existing_edge.get("id"),),
                    fallback_kwargs={},
                )

        for op in temporal.update_ops:
            if op.get("type") == "supersede":
                await self._memory.semantic.mark_superseded(
                    edge_id=op["edge_id"],
                    valid_to=op["valid_to"],
                )

        for edge in temporal.new_edges:
            await _call_with_fallback(
                self._memory.staging.encode_edge,
                rich_kwargs={
                    "workspace_id": workspace_id,
                    "source_node": edge["source"],
                    "relation": edge["relation"],
                    "target_node": edge["target"],
                    "target_workspace_id": edge.get("target_workspace_id"),
                    "source_ref": source["id"],
                    "confidence": edge.get("confidence", "EXTRACTED"),
                    "epistemic_source": edge.get("epistemic_source", "asserted"),
                    "importance_score": edge.get("importance_score", 0.5),
                    "never_decay": bool(edge.get("never_decay", 0)),
                    "valid_from": edge.get("valid_from"),
                    "temporal_ambiguous": edge.get("temporal_ambiguous", False),
                },
                fallback_args=(
                    edge["source"],
                    edge["relation"],
                    edge["target"],
                    interaction_id,
                    session_id,
                ),
                fallback_kwargs={
                    "workspace_id": workspace_id,
                    "importance_score": float(edge.get("importance_score", 0.5)),
                    "never_decay": bool(edge.get("never_decay", 0)),
                },
            )

        for contradiction in validated.contradictions:
            await self._memory.contradictions.detect(
                workspace_id=workspace_id,
                subject=contradiction["subject"],
                key_or_relation=contradiction["key"],
                value_a=contradiction["value_a"],
                value_b=contradiction["value_b"],
                value_a_source=contradiction.get("existing_edge_id", ""),
                value_b_source=source["id"],
                value_a_confidence=contradiction.get("value_a_confidence", "EXTRACTED"),
                value_b_confidence=contradiction.get("value_b_confidence", "EXTRACTED"),
                core_node_set=(
                    [contradiction["subject"]]
                    if contradiction.get("severity") == "severe"
                    else None
                ),
            )

        for question in validated.open_questions:
            await self._memory.open_questions.add_question(
                workspace_id=workspace_id,
                question=question["question"],
                raised_by=_normalize_raised_by(question.get("raised_by", "")),
                severity=question.get("severity", "moderate"),
                context_node=question.get("context_node", ""),
                context_input=question.get("context_input", ""),
            )

        for proposal in extracted.new_type_proposals:
            await self._persist_type_proposal(workspace_id, proposal)

        if extracted.narrative_chunk:
            await _call_with_fallback(
                self._memory.staging.encode,
                rich_kwargs={
                    "content": extracted.narrative_chunk,
                    "entities": {"nodes": [node.get("label") for node in extracted.nodes]},
                    "workspace_id": workspace_id,
                    "source_id": source["id"],
                    "event_type": triage.input_kinds[0] if triage.input_kinds else "unknown",
                    "importance_score": _avg_importance(temporal.new_edges),
                    "never_decay": int(
                        any(int(edge.get("never_decay", 0)) == 1 for edge in temporal.new_edges)
                    ),
                },
                fallback_args=(),
                fallback_kwargs={
                    "content": extracted.narrative_chunk,
                    "entities": {
                        "nodes": [node.get("label") for node in extracted.nodes],
                        "source_id": source["id"],
                        "input_kinds": list(triage.input_kinds),
                        "edge_count": len(temporal.new_edges),
                        "importance_score": _avg_importance(temporal.new_edges),
                        "never_decay": int(
                            any(
                                int(edge.get("never_decay", 0)) == 1
                                for edge in temporal.new_edges
                            )
                        ),
                    },
                    "interaction_id": interaction_id,
                    "session_id": session_id,
                    "workspace_id": workspace_id,
                },
            )

    async def _persist_type_proposal(self, workspace_id: str, proposal: dict) -> None:
        kind = proposal.get("kind", "node")
        confidence = proposal.get("confidence", "AMBIGUOUS")
        name = proposal["name"]
        definition = _definition_dict(proposal.get("definition", {}))

        if confidence == "EXTRACTED":
            if kind in {"relation", "relation_type"} and hasattr(
                self._memory.ontology,
                "register_relation_type",
            ):
                await self._memory.ontology.register_relation_type(
                    workspace_id,
                    name,
                    source_snippet=proposal.get("source_snippet", ""),
                )
            else:
                await self._memory.ontology.register_node_type(
                    workspace_id,
                    name,
                    schema=definition.get("schema") if isinstance(definition, dict) else {},
                    source_snippet=proposal.get("source_snippet", ""),
                )
            return

        if kind in {"relation", "relation_type"} and hasattr(
            self._memory.ontology,
            "propose_relation_type",
        ):
            await self._memory.ontology.propose_relation_type(
                workspace_id,
                name,
                definition=definition,
                confidence=confidence,
                source_input=proposal.get("source_snippet", ""),
            )
        else:
            await self._memory.ontology.propose_node_type(
                workspace_id,
                name,
                definition=definition,
                confidence=confidence,
                source_input=proposal.get("source_snippet", ""),
            )


async def _call_with_fallback(
    fn,
    rich_kwargs: dict,
    fallback_args: tuple,
    fallback_kwargs: dict,
):
    try:
        return await fn(**rich_kwargs)
    except TypeError:
        return await fn(*fallback_args, **fallback_kwargs)


def _normalize_raised_by(value: str) -> str:
    allowed = {"ambiguity_detector", "unknown_fact", "contradiction", "user"}
    return value if value in allowed else "ambiguity_detector"


def _definition_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"description": str(value)}


def _avg_importance(edges: list[dict]) -> float:
    if not edges:
        return 0.5
    vals = [float(edge.get("importance_score", 0.5)) for edge in edges]
    return sum(vals) / len(vals)
