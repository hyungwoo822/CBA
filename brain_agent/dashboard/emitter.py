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
