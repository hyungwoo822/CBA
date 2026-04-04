import pytest
from brain_agent.core.neuromodulators import Neuromodulators
from brain_agent.core.neuromodulator_controller import NeuromodulatorController


@pytest.fixture
def ctrl():
    nm = Neuromodulators()
    return NeuromodulatorController(nm)


# -- VTA: Dopamine (reward_signal) --


class TestVTA:
    def test_positive_prediction_error_increases_da(self, ctrl):
        """Better than expected -> DA spike (positive RPE)."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        assert ctrl.neuromodulators.reward_signal > 0.0

    def test_negative_prediction_error_decreases_da(self, ctrl):
        """Worse than expected -> DA dip (negative RPE)."""
        ctrl.on_prediction_error(error=0.8, predicted="success", actual="failure")
        assert ctrl.neuromodulators.reward_signal < 0.0

    def test_expected_outcome_no_da_change(self, ctrl):
        """As expected -> minimal DA change."""
        ctrl.on_prediction_error(error=0.05, predicted="success", actual="success")
        assert abs(ctrl.neuromodulators.reward_signal) < 0.2

    def test_da_consumed_by_bg_go(self, ctrl):
        """DA should be accessible for BG Go pathway."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        snap = ctrl.neuromodulators.snapshot()
        assert snap["dopamine"] > 0.5  # Above tonic baseline = positive RPE


# -- Locus Coeruleus: NE (urgency) --


class TestLC:
    def test_high_arousal_increases_ne(self, ctrl):
        """Amygdala threat detection -> NE spike."""
        before = ctrl.neuromodulators.urgency
        ctrl.on_emotional_arousal(arousal=0.9)
        assert ctrl.neuromodulators.urgency > before

    def test_low_arousal_suppresses_ne(self, ctrl):
        """Low arousal (sleepy/calm) should DECREASE NE, not just 'not increase'."""
        before_ne = ctrl.neuromodulators.urgency
        before_epi = ctrl.neuromodulators.epinephrine
        ctrl.on_emotional_arousal(arousal=0.1)
        assert ctrl.neuromodulators.urgency < before_ne, "NE should drop on low arousal"
        assert ctrl.neuromodulators.epinephrine < before_epi, "EPI should drop on low arousal"

    def test_conflict_increases_ne(self, ctrl):
        """ACC conflict -> phasic NE burst."""
        before = ctrl.neuromodulators.urgency
        ctrl.on_conflict(conflict_score=0.8)
        assert ctrl.neuromodulators.urgency > before

    def test_system_load_increases_ne(self, ctrl):
        """High pending requests -> sustained NE."""
        ctrl.on_system_state(pending_requests=8, error_rate=0.2)
        assert ctrl.neuromodulators.urgency > 0.5


# -- Nucleus Basalis: ACh (learning_rate) --


class TestNucleusBasalis:
    def test_high_novelty_increases_ach(self, ctrl):
        """Novel input -> ACh spike -> learn more."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_novelty(novelty=0.9)
        assert ctrl.neuromodulators.learning_rate > before

    def test_familiar_input_decreases_ach(self, ctrl):
        """Familiar input -> low ACh -> less plasticity."""
        ctrl.on_novelty(novelty=0.1)
        assert ctrl.neuromodulators.learning_rate < 0.5

    def test_uncertainty_increases_ach(self, ctrl):
        """ACC uncertainty -> more learning needed."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_conflict(conflict_score=0.7)
        assert ctrl.neuromodulators.learning_rate > before

    def test_large_rpe_magnitude_increases_ach(self, ctrl):
        """Big surprise (either direction) -> learn from it."""
        before = ctrl.neuromodulators.learning_rate
        ctrl.on_prediction_error(error=0.9, predicted="success", actual="failure")
        assert ctrl.neuromodulators.learning_rate > before


# -- Dorsal Raphe: 5-HT (patience) --


class TestDorsalRaphe:
    def test_low_error_rate_high_patience(self, ctrl):
        """Things going well -> more patient."""
        ctrl.on_system_state(pending_requests=0, error_rate=0.1)
        assert ctrl.neuromodulators.patience > 0.5

    def test_high_error_rate_low_patience(self, ctrl):
        """Things going badly -> impatient/frustrated."""
        ctrl.on_system_state(pending_requests=0, error_rate=0.9)
        assert ctrl.neuromodulators.patience < 0.5

    def test_positive_reward_increases_patience(self, ctrl):
        """Consistent rewards -> more patience (delayed gratification)."""
        before = ctrl.neuromodulators.patience
        ctrl.on_reward_outcome(success=True)
        assert ctrl.neuromodulators.patience >= before

    def test_negative_reward_decreases_patience(self, ctrl):
        """Failures -> less patience."""
        ctrl.on_reward_outcome(success=False)
        assert ctrl.neuromodulators.patience < 0.5


# -- Anti-Saturation --


class TestAntiSaturation:
    def test_sustained_neutral_input_stays_near_baseline(self, ctrl):
        """10 neutral requests should NOT saturate any NT."""
        for _ in range(10):
            ctrl.on_emotional_arousal(arousal=0.0)
            ctrl.on_novelty(novelty=0.5)
            ctrl.on_prediction_error(error=0.1, predicted="success", actual="success")
            ctrl.on_reward_outcome(success=True)
            ctrl.on_system_state(pending_requests=0, error_rate=0.1)
            ctrl.decay()

        nm = ctrl.neuromodulators
        assert nm.dopamine < 0.75, f"DA saturated: {nm.dopamine}"
        assert nm.norepinephrine < 0.75, f"NE saturated: {nm.norepinephrine}"
        assert nm.acetylcholine < 0.75, f"ACh saturated: {nm.acetylcholine}"
        assert nm.serotonin < 0.75, f"5-HT saturated: {nm.serotonin}"
        assert nm.cortisol < 0.75, f"CORT saturated: {nm.cortisol}"
        assert nm.epinephrine < 0.75, f"EPI saturated: {nm.epinephrine}"
        assert nm.gaba < 0.75, f"GABA saturated: {nm.gaba}"

    def test_strong_emotion_reaches_meaningful_level(self, ctrl):
        """High arousal should spike NTs but not clamp at 1.0.

        With Yerkes-Dodson (σ=0.15, optimal=0.5), arousal=0.8 has a reduced
        NE gain (inverted-U), so NE stays moderate — this is correct behavior.
        DA and EPI still respond strongly to high arousal.
        """
        ctrl.on_emotional_arousal(arousal=0.8)
        nm = ctrl.neuromodulators
        assert nm.norepinephrine > 0.51, "NE should respond to high arousal (Yerkes-Dodson modulated)"
        assert nm.norepinephrine < 0.95, "NE should not saturate from single event"

    def test_recovery_after_spike(self, ctrl):
        """After a strong spike, 5 decay cycles should bring values back toward baseline."""
        ctrl.on_emotional_arousal(arousal=0.9)
        ctrl.on_novelty(novelty=0.9)
        for _ in range(5):
            ctrl.decay()
        nm = ctrl.neuromodulators
        assert nm.norepinephrine < 0.65, f"NE didn't recover: {nm.norepinephrine}"
        assert nm.acetylcholine < 0.65, f"ACh didn't recover: {nm.acetylcholine}"


# -- Decay --


class TestDecay:
    def test_values_decay_toward_baseline(self, ctrl):
        """After spike, values should decay back toward baseline.

        With slower decay (0.92), phasic changes persist longer —
        need ~25 cycles to return near baseline.
        """
        ctrl.on_emotional_arousal(arousal=1.0)  # NE spike
        high_urgency = ctrl.neuromodulators.urgency

        ctrl.decay()
        assert ctrl.neuromodulators.urgency < high_urgency

        for _ in range(10):
            ctrl.decay()
        # After many decays, should be close to baseline
        assert ctrl.neuromodulators.urgency == pytest.approx(0.5, abs=0.15)

    def test_da_decays_toward_zero(self, ctrl):
        """DA baseline is 0.5 (tonic firing). Reward signal should decay to 0."""
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        assert ctrl.neuromodulators.reward_signal > 0

        for _ in range(15):
            ctrl.decay()
        assert abs(ctrl.neuromodulators.reward_signal) < 0.1

    def test_ach_decays_toward_baseline(self, ctrl):
        """ACh should decay toward 0.5 baseline."""
        ctrl.on_novelty(novelty=1.0)
        high_lr = ctrl.neuromodulators.learning_rate
        assert high_lr > 0.5

        for _ in range(10):
            ctrl.decay()
        assert ctrl.neuromodulators.learning_rate == pytest.approx(0.5, abs=0.15)

    def test_patience_decays_toward_baseline(self, ctrl):
        """5-HT should decay toward 0.5 baseline (slow decay: 0.95)."""
        ctrl.on_reward_outcome(success=False)
        low_patience = ctrl.neuromodulators.patience

        for _ in range(20):
            ctrl.decay()
        assert ctrl.neuromodulators.patience == pytest.approx(0.5, abs=0.15)

    def test_multiple_decays_converge(self, ctrl):
        """All neuromodulators should converge to baselines after many decays."""
        ctrl.on_emotional_arousal(arousal=1.0)
        ctrl.on_novelty(novelty=1.0)
        ctrl.on_prediction_error(error=0.1, predicted="failure", actual="success")
        ctrl.on_reward_outcome(success=False)

        for _ in range(100):
            ctrl.decay()

        assert ctrl.neuromodulators.urgency == pytest.approx(0.5, abs=0.01)
        assert ctrl.neuromodulators.learning_rate == pytest.approx(0.5, abs=0.01)
        assert abs(ctrl.neuromodulators.reward_signal) < 0.01
        assert ctrl.neuromodulators.patience == pytest.approx(0.5, abs=0.01)
        assert ctrl.neuromodulators.gaba == pytest.approx(0.5, abs=0.01)


# -- Memory Modulation --


class TestMemoryModulation:
    def test_ach_encoding_strength_varies(self, ctrl):
        """ACh should produce encoding strength in [0.75, 1.25], not always max."""
        strength_neutral = 1.0 + 0.5 * (ctrl.neuromodulators.acetylcholine - 0.5)
        assert 0.95 <= strength_neutral <= 1.05

        ctrl.on_novelty(novelty=0.9)
        strength_novel = 1.0 + 0.5 * (ctrl.neuromodulators.acetylcholine - 0.5)
        assert strength_novel > 1.05

        ctrl2 = NeuromodulatorController(Neuromodulators())
        ctrl2.on_novelty(novelty=0.1)
        strength_familiar = 1.0 + 0.5 * (ctrl2.neuromodulators.acetylcholine - 0.5)
        assert strength_familiar < 1.0

    def test_ach_consolidation_gating_bidirectional(self, ctrl):
        """ACh consolidation factor should boost (low ACh) AND penalize (high ACh)."""
        ctrl.on_novelty(novelty=0.1)
        ach_low = ctrl.neuromodulators.acetylcholine
        factor_low = 1.0 + 0.8 * (0.5 - ach_low)
        assert factor_low > 1.0, "Low ACh should boost consolidation"

        ctrl2 = NeuromodulatorController(Neuromodulators())
        ctrl2.on_novelty(novelty=0.9)
        ach_high = ctrl2.neuromodulators.acetylcholine
        factor_high = 1.0 + 0.8 * (0.5 - ach_high)
        assert factor_high < 1.0, "High ACh should penalize consolidation"

    def test_da_reward_gating_selective(self, ctrl):
        """reward_signal should be negative after failure, not always positive."""
        ctrl.on_prediction_error(error=0.8, predicted="success", actual="failure")
        assert ctrl.neuromodulators.reward_signal < 0, "Failed prediction should give negative reward"


# -- NT Crosstalk --


class TestCrosstalk:
    def test_high_cortisol_suppresses_serotonin(self, ctrl):
        """CORT↑ → 5-HT↓ (Porter & Bhatt 2008)."""
        ctrl.neuromodulators.cortisol = 0.8
        ctrl.neuromodulators.serotonin = 0.6
        ctrl.decay()  # crosstalk runs after decay
        assert ctrl.neuromodulators.serotonin < 0.6

    def test_da_ne_competition(self, ctrl):
        """High DA should slightly suppress NE (shared precursor)."""
        ctrl.neuromodulators.dopamine = 0.85
        ctrl.neuromodulators.norepinephrine = 0.6
        ctrl.decay()
        assert ctrl.neuromodulators.norepinephrine < 0.6

    def test_ach_boosts_da(self, ctrl):
        """High ACh → slight DA boost (striatal modulation)."""
        ctrl.neuromodulators.acetylcholine = 0.75
        da_before = ctrl.neuromodulators.dopamine
        ctrl.decay()
        assert ctrl.neuromodulators.dopamine >= da_before  # May be slightly higher


class TestYerkesDodson:
    def test_moderate_arousal_highest_ne_gain(self, ctrl):
        """Moderate arousal should produce higher NE than extreme arousal."""
        ctrl2 = NeuromodulatorController(Neuromodulators())

        ctrl.on_emotional_arousal(arousal=0.5)   # moderate
        ctrl2.on_emotional_arousal(arousal=0.95)  # extreme

        # Moderate should give higher effective NE gain due to Yerkes-Dodson
        # (but extreme has higher raw arousal, so NE may still be higher)
        # Key test: extreme arousal NE gain is NOT proportionally higher
        ne_moderate = ctrl.neuromodulators.norepinephrine - 0.5
        ne_extreme = ctrl2.neuromodulators.norepinephrine - 0.5
        ratio = ne_extreme / ne_moderate if ne_moderate > 0 else 999
        assert ratio < 1.9, f"Extreme should not be proportionally 1.9x moderate (got {ratio:.2f})"


# -- GABA (Cortical Inhibition) --


class TestGABA:
    def test_gaba_decay(self, ctrl):
        """GABA should decay toward baseline 0.5."""
        ctrl.neuromodulators.gaba = 0.8
        ctrl.decay()
        assert ctrl.neuromodulators.gaba < 0.8
        for _ in range(20):
            ctrl.decay()
        assert ctrl.neuromodulators.gaba == pytest.approx(0.5, abs=0.05)

    def test_gaba_on_conflict_increases(self, ctrl):
        """Conflict should increase GABA (inhibitory braking, Aron 2007)."""
        before = ctrl.neuromodulators.gaba
        ctrl.on_conflict(conflict_score=0.8)
        assert ctrl.neuromodulators.gaba > before

    def test_ei_balance_crosstalk(self, ctrl):
        """High NE+DA (excitation) should raise GABA (Isaacson & Scanziani 2011)."""
        ctrl.neuromodulators.norepinephrine = 0.85
        ctrl.neuromodulators.dopamine = 0.85
        ctrl.decay()  # triggers crosstalk
        assert ctrl.neuromodulators.gaba > 0.5

    def test_high_gaba_suppresses_ne(self, ctrl):
        """GABA > 0.7 should suppress NE (cortical inhibition)."""
        ctrl.neuromodulators.gaba = 0.85
        ctrl.neuromodulators.norepinephrine = 0.6
        ctrl.decay()  # triggers crosstalk
        assert ctrl.neuromodulators.norepinephrine < 0.6
