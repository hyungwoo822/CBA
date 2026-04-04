from brain_agent.core.neuromodulators import Neuromodulators


def test_default_values():
    nm = Neuromodulators()
    # Primary NT names
    assert nm.dopamine == 0.5
    assert nm.norepinephrine == 0.5
    assert nm.serotonin == 0.5
    assert nm.acetylcholine == 0.5
    assert nm.cortisol == 0.5
    assert nm.epinephrine == 0.5
    # Backward-compat aliases
    assert nm.urgency == 0.5
    assert nm.learning_rate == 0.5
    assert nm.patience == 0.5
    assert nm.reward_signal == 0.0  # (dopamine 0.5 → reward_signal 0.0)


def test_clamp_values():
    nm = Neuromodulators()
    nm.norepinephrine = 1.5
    assert nm.norepinephrine == 1.0
    nm.norepinephrine = -0.5
    assert nm.norepinephrine == 0.0
    # Aliases also clamp
    nm.urgency = 1.5
    assert nm.urgency == 1.0
    # New hormones clamp
    nm.cortisol = 2.0
    assert nm.cortisol == 1.0
    nm.epinephrine = -1.0
    assert nm.epinephrine == 0.0


def test_alias_bidirectional():
    """Aliases and NT names share the same backing store."""
    nm = Neuromodulators()
    nm.urgency = 0.8
    assert nm.norepinephrine == 0.8
    nm.norepinephrine = 0.3
    assert nm.urgency == 0.3

    nm.learning_rate = 0.9
    assert nm.acetylcholine == 0.9

    nm.patience = 0.2
    assert nm.serotonin == 0.2


def test_reward_signal_maps_to_dopamine():
    """reward_signal [-1,1] maps to dopamine [0,1]."""
    nm = Neuromodulators()
    nm.reward_signal = 1.0   # max positive RPE
    assert nm.dopamine == 1.0
    nm.reward_signal = -1.0  # max negative RPE
    assert nm.dopamine == 0.0
    nm.reward_signal = 0.0   # neutral
    assert nm.dopamine == 0.5


def test_update():
    nm = Neuromodulators()
    nm.update(urgency=0.8, patience=0.3)
    assert nm.norepinephrine == 0.8
    assert nm.serotonin == 0.3


def test_snapshot():
    nm = Neuromodulators()
    nm.update(urgency=0.7)
    snap = nm.snapshot()
    assert snap["norepinephrine"] == 0.7
    assert snap["dopamine"] == 0.5
    assert snap["cortisol"] == 0.5
    assert snap["epinephrine"] == 0.5
    assert snap["gaba"] == 0.5
    assert isinstance(snap, dict)
    assert len(snap) == 7


# -- GABA --


def test_gaba_default():
    nm = Neuromodulators()
    assert nm.gaba == 0.5


def test_gaba_clamp():
    nm = Neuromodulators()
    nm.gaba = 1.5
    assert nm.gaba == 1.0
    nm.gaba = -0.3
    assert nm.gaba == 0.0


def test_gaba_in_snapshot():
    nm = Neuromodulators()
    nm.gaba = 0.75
    snap = nm.snapshot()
    assert snap["gaba"] == 0.75


def test_gaba_load_from():
    nm = Neuromodulators()
    nm.load_from({"gaba": 0.8, "dopamine": 0.6})
    assert nm.gaba == 0.8
    assert nm.dopamine == 0.6


def test_gaba_load_from_missing_defaults():
    nm = Neuromodulators()
    nm.load_from({})  # no gaba key
    assert nm.gaba == 0.5


def test_inhibition_alias():
    """inhibition alias is bidirectional with gaba."""
    nm = Neuromodulators()
    nm.inhibition = 0.9
    assert nm.gaba == 0.9
    nm.gaba = 0.3
    assert nm.inhibition == 0.3
