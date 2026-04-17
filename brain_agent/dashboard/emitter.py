"""Pipeline event emitter -- hooks into brain regions to emit dashboard events."""
from __future__ import annotations

from brain_agent.dashboard.server import event_bus


class DashboardEmitter:
    """Emits events from pipeline processing to the dashboard EventBus."""

    async def region_activation(self, region: str, level: float, mode: str = "active") -> None:
        await event_bus.emit("region_activation", {
            "region": region, "level": level, "mode": mode
        })

    async def network_switch(self, from_mode: str, to_mode: str, trigger: str = "") -> None:
        await event_bus.emit("network_switch", {
            "from": from_mode, "to": to_mode, "trigger": trigger
        })

    async def routing_event(self, source: str, targets: list[str], signal_type: str, label: str = "") -> None:
        await event_bus.emit("routing_event", {
            "source": source, "targets": targets, "signal_type": signal_type, "label": label
        })

    async def memory_event(self, event_type: str, store: str, memory_id: str = "") -> None:
        await event_bus.emit("memory_event", {
            "type": event_type, "store": store, "id": memory_id
        })

    async def memory_flow(self, sensory: int, working: int, staging: int, episodic: int, semantic: int, procedural: int = 0) -> None:
        await event_bus.emit("memory_flow", {
            "sensory": sensory, "working": working, "staging": staging,
            "episodic": episodic, "semantic": semantic, "procedural": procedural,
        })

    async def neuromodulator_update(self, **levels) -> None:
        """Emit all 6 neurotransmitter levels to dashboard."""
        await event_bus.emit("neuromodulator", levels)

    async def signal_flow(self, source: str, target: str, signal_type: str, strength: float, data_summary: dict | None = None) -> None:
        payload = {
            "source": source, "target": target,
            "signal_type": signal_type, "strength": strength,
        }
        if data_summary:
            payload["data_summary"] = data_summary
        await event_bus.emit("signal_flow", payload)

    async def region_io(self, region: str, input_summary: dict, output_summary: dict, processing: str) -> None:
        await event_bus.emit("region_io", {
            "region": region,
            "input": input_summary,
            "output": output_summary,
            "processing": processing,
        })

    async def region_processing(self, region: str, phase: str, summary: str, details: dict | None = None) -> None:
        """Emit real-time processing status from a brain region to the chat stream.

        This allows the dashboard to show what each region is "thinking" as it processes.
        """
        payload = {"region": region, "phase": phase, "summary": summary}
        if details:
            payload["details"] = details
        await event_bus.emit("region_processing", payload)

    async def broadcast(self, content: str, origin: str) -> None:
        await event_bus.emit("broadcast", {"content": content, "origin": origin})

    async def knowledge_update(
        self, node_count: int, edge_count: int, community_count: int,
    ) -> None:
        """Emit knowledge graph structural update."""
        await event_bus.emit("knowledge_update", {
            "nodes": node_count,
            "edges": edge_count,
            "communities": community_count,
        })

    async def knowledge_diff(self, diff: dict) -> None:
        """Emit neuroplasticity diff (LTP/LTD/synaptogenesis/pruning)."""
        await event_bus.emit("knowledge_diff", diff)

    async def workspace_changed(
        self,
        workspace_id: str,
        workspace_name: str,
        session_id: str,
    ) -> None:
        await event_bus.emit("workspace_changed", {
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            "session_id": session_id,
        })

    async def clarification_requested(
        self,
        question_id: str | dict = "",
        question: str = "",
        severity: str = "moderate",
        workspace_id: str = "",
        context_input: str = "",
        raised_by: str = "pipeline",
        **extra,
    ) -> None:
        """Emit extraction-time clarification questions."""
        if isinstance(question_id, dict):
            payload = question_id
        else:
            payload = {
                "id": question_id,
                "question_id": question_id,
                "question": question,
                "severity": severity,
                "workspace_id": workspace_id,
                "context_input": context_input,
                "raised_by": raised_by,
                **extra,
            }
        await event_bus.emit("clarification_requested", payload)

    async def contradiction_detected(
        self,
        contradiction_id: str,
        subject: str,
        value_a: str,
        value_b: str,
        severity: str,
        workspace_id: str,
        **extra,
    ) -> None:
        await event_bus.emit("contradiction_detected", {
            "id": contradiction_id,
            "contradiction_id": contradiction_id,
            "subject": subject,
            "value_a": value_a,
            "value_b": value_b,
            "severity": severity,
            "workspace_id": workspace_id,
            **extra,
        })

    async def ontology_proposal(
        self,
        proposal_id: str,
        kind: str,
        proposed_name: str,
        confidence: str,
        workspace_id: str,
        source_snippet: str = "",
        **extra,
    ) -> None:
        await event_bus.emit("ontology_proposal", {
            "id": proposal_id,
            "proposal_id": proposal_id,
            "kind": kind,
            "proposed_name": proposed_name,
            "confidence": confidence,
            "workspace_id": workspace_id,
            "source_snippet": source_snippet,
            **extra,
        })

    async def question_answered(self, question_id: str, workspace_id: str) -> None:
        await event_bus.emit("question_answered", {
            "question_id": question_id,
            "workspace_id": workspace_id,
        })

    async def contradiction_resolved(
        self,
        contradiction_id: str,
        resolution: str,
        workspace_id: str,
    ) -> None:
        await event_bus.emit("contradiction_resolved", {
            "contradiction_id": contradiction_id,
            "resolution": resolution,
            "workspace_id": workspace_id,
        })

    async def proposal_decided(
        self,
        proposal_id: str,
        status: str,
        workspace_id: str,
    ) -> None:
        await event_bus.emit("proposal_decided", {
            "proposal_id": proposal_id,
            "status": status,
            "workspace_id": workspace_id,
        })
