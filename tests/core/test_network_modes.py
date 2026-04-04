from brain_agent.core.network_modes import NetworkMode, TripleNetworkController


def test_default_mode_is_dmn():
    ctrl = TripleNetworkController()
    assert ctrl.current_mode == NetworkMode.DMN


def test_switch_to_ecn():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    assert ctrl.current_mode == NetworkMode.ECN


def test_active_regions_differ_by_mode():
    ctrl = TripleNetworkController()
    dmn_regions = ctrl.get_active_regions()
    ctrl.switch_to(NetworkMode.ECN)
    ecn_regions = ctrl.get_active_regions()
    assert dmn_regions != ecn_regions


def test_ecn_regions():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    active = ctrl.get_active_regions()
    assert "prefrontal_cortex" in active
    assert "acc" in active
    assert "basal_ganglia" in active
    assert "cerebellum" in active


def test_dmn_regions():
    ctrl = TripleNetworkController()
    active = ctrl.get_active_regions()
    assert "hippocampus" in active
    assert "prefrontal_cortex" not in active


def test_switch_emits_history():
    ctrl = TripleNetworkController()
    ctrl.switch_to(NetworkMode.ECN)
    ctrl.switch_to(NetworkMode.DMN)
    assert len(ctrl.switch_history) == 2
