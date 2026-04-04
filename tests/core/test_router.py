import pytest
from brain_agent.core.router import ThalamicRouter
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController
from brain_agent.core.neuromodulators import Neuromodulators

@pytest.fixture
def router():
    return ThalamicRouter(network_ctrl=TripleNetworkController(), neuromodulators=Neuromodulators())

def test_route_plan_in_ecn(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"content": "test"})
    targets = router.resolve_targets(sig)
    assert "acc" in targets
    assert "basal_ganglia" in targets

def test_plan_suppressed_in_dmn(router):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={})
    targets = router.resolve_targets(sig)
    assert len(targets) == 0

def test_broadcast_goes_to_all(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.GWT_BROADCAST, source="workspace", payload={})
    targets = router.resolve_targets(sig)
    assert len(targets) > 3

def test_priority_boosted_by_arousal(router):
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={},
                 emotional_tag=EmotionalTag(valence=-0.5, arousal=0.9))
    p = router.compute_priority(sig)
    sig_neutral = Signal(type=SignalType.PLAN, source="pfc", payload={})
    p_neutral = router.compute_priority(sig_neutral)
    assert p > p_neutral

def test_conflict_forces_ecn_in_dmn(router):
    sig = Signal(type=SignalType.CONFLICT_DETECTED, source="acc", payload={})
    targets = router.resolve_targets(sig)
    assert "prefrontal_cortex" in targets

def test_routing_events_emitted(router):
    router._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={})
    router.resolve_targets(sig)
    assert len(router.event_log) == 1
