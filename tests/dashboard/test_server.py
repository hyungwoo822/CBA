# tests/dashboard/test_server.py
import pytest
from brain_agent.dashboard.server import EventBus, DashboardEvent, create_app


async def test_event_bus_emit():
    bus = EventBus()
    await bus.emit("test", {"key": "value"})
    recent = bus.get_recent(1)
    assert len(recent) == 1
    assert recent[0]["type"] == "test"


async def test_event_bus_buffer_limit():
    bus = EventBus(buffer_size=5)
    for i in range(10):
        await bus.emit("test", {"i": i})
    recent = bus.get_recent(10)
    assert len(recent) == 5
    assert recent[0]["payload"]["i"] == 5  # oldest kept is 5


async def test_dashboard_event_json():
    evt = DashboardEvent(event_type="region_activation", payload={"region": "pfc", "level": 0.8})
    j = evt.to_json()
    import json
    parsed = json.loads(j)
    assert parsed["type"] == "region_activation"
    assert parsed["payload"]["region"] == "pfc"


def test_create_app():
    app = create_app()
    assert app.title == "Brain Agent Dashboard"


async def test_emitter():
    from brain_agent.dashboard.emitter import DashboardEmitter
    emitter = DashboardEmitter()
    await emitter.region_activation("pfc", 0.9, "high_activity")
    await emitter.network_switch("DMN", "ECN", "salience")
    await emitter.memory_flow(1, 3, 12, 847, 5)
    await emitter.neuromodulator_update(
        dopamine=0.5, norepinephrine=0.5, serotonin=0.5,
        acetylcholine=0.5, cortisol=0.5, epinephrine=0.5,
    )
    # Should not raise
