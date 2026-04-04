import pytest
from unittest.mock import AsyncMock, patch
from brain_agent.dashboard.emitter import DashboardEmitter


async def test_signal_flow_emits_event():
    emitter = DashboardEmitter()
    with patch("brain_agent.dashboard.emitter.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        await emitter.signal_flow("thalamus", "amygdala", "EXTERNAL_INPUT", 0.8)
        mock_bus.emit.assert_awaited_once_with("signal_flow", {
            "source": "thalamus",
            "target": "amygdala",
            "signal_type": "EXTERNAL_INPUT",
            "strength": 0.8,
        })
