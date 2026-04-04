"""Neuromodulator Dynamics Controller.

Models 7 neurochemical systems with brain-faithful update triggers:

  VTA/SNc (Dopamine)       -> dopamine       -- Reward Prediction Error
  Locus Coeruleus (NE)     -> norepinephrine -- Arousal, conflict, system load
  Nucleus Basalis (ACh)    -> acetylcholine  -- Novelty, uncertainty, surprise
  Dorsal Raphe (5-HT)      -> serotonin      -- Error rate, reward history
  Adrenal Cortex (CORT)    -> cortisol       -- HPA axis stress response
  Adrenal Medulla (EPI)    -> epinephrine    -- Sympathetic fight-or-flight
  Cortical Interneurons    -> gaba           -- Inhibitory tone, E/I balance

Each nucleus receives inputs from specific brain regions and modulates
downstream targets. Values drift toward baseline between events via
exponential decay (with different time constants per system).

References:
  - Schultz (1997): DA and reward prediction error
  - Grace (2000): Tonic vs phasic DA firing
  - Aston-Jones & Cohen (2005): LC-NE and adaptive gain
  - Hasselmo (2006): ACh and novelty/learning
  - Doya (2002): Serotonin and temporal discounting
  - Sapolsky (2004): HPA axis, cortisol and stress
  - McEwen (2007): Allostatic load from chronic stress
  - Cannon (1929): Fight-or-flight and epinephrine
  - Cahill & McGaugh (1998): Epinephrine enhances memory consolidation
  - de Quervain et al. (2000): Cortisol impairs memory retrieval
  - Dickerson & Kemeny (2004): Social-evaluative threat → HPA
  - Buzsaki (2006): GABAergic interneurons and cortical oscillations
  - Isaacson & Scanziani (2011): E/I balance homeostatic compensation
  - Aron (2007): Conflict → inhibitory braking (GABA)
"""
from __future__ import annotations

from brain_agent.core.neuromodulators import Neuromodulators

# ── Baselines (resting state = tonic firing rate) ────────────────
DA_BASELINE = 0.5          # Tonic DA firing (Grace 2000)
NE_BASELINE = 0.5          # Moderate alertness
ACH_BASELINE = 0.5         # Moderate plasticity
SEROTONIN_BASELINE = 0.5   # Moderate patience
CORT_BASELINE = 0.5        # No stress
EPI_BASELINE = 0.5         # No arousal

# ── Decay rates (per interaction cycle, toward baseline) ─────────
# Different time constants per neurochemical system:
#   Catecholamines (DA, NE): moderate (~seconds in real brain)
#   ACh: moderate
#   5-HT: slow (mood/patience shifts gradually)
#   Cortisol: SLOW (half-life ~60-90 min, McEwen 2007)
#   Epinephrine: FAST (rapidly metabolized, Cannon 1929)
#
# NOTE: Values below 1.0 drift toward baseline each cycle.
# 0.92 = slow drift (state persists across ~10 interactions)
# 0.85 = moderate drift (~5 interactions to half-life)
# 0.75 = fast drift (~2-3 interactions)
DECAY_RATE = 0.85          # DA,NE,ACh: half-life ~5 cycles
SEROTONIN_DECAY_RATE = 0.90  # 5-HT: half-life ~7 cycles
CORT_DECAY_RATE = 0.93     # Cortisol: half-life ~10 cycles
EPI_DECAY_RATE = 0.75      # EPI: fast, ~2.5 cycles

# ── Phasic response gains ────────────────────────────────────────
# These are ADDITIVE deltas. A gain of 0.15 on a 0.5 baseline means
# a full-strength trigger moves the value to 0.65 in one cycle.
# Gains are tuned so that typical inputs produce visible movement
# (±0.05 to ±0.20) on the dashboard.

# DA (VTA/SNc) — Reward Prediction Error (Schultz 1997)
DA_GAIN = 0.4              # Prediction error magnitude → DA

# NE (Locus Coeruleus) — Arousal + Conflict + Load (Aston-Jones & Cohen 2005)
NE_AROUSAL_GAIN = 0.25     # Arousal → NE (emotional intensity drives alertness)
NE_CONFLICT_GAIN = 0.15    # Conflict → NE (uncertainty demands attention)
NE_LOAD_GAIN = 0.08        # System load → NE

# ACh (Nucleus Basalis) — Novelty + Uncertainty + Surprise (Hasselmo 2006)
ACH_NOVELTY_GAIN = 0.3     # Novelty → ACh (novel stimuli boost encoding)
ACH_UNCERTAINTY_GAIN = 0.12  # Conflict/uncertainty → ACh
ACH_SURPRISE_GAIN = 0.15   # |RPE| → ACh (surprise = learn)

# 5-HT (Dorsal Raphe) — Error Rate + Reward History (Doya 2002)
SEROTONIN_ERROR_GAIN = 0.10   # Error rate → 5-HT (inverse: more errors = less patience)
SEROTONIN_REWARD_GAIN = 0.02  # Success/failure nudge (visible per interaction)

# CORT (Adrenal Cortex — HPA axis)
CORT_STRESS_GAIN = 0.10       # High arousal → cortisol (Dickerson & Kemeny 2004)
CORT_ERROR_GAIN = 0.08        # Large prediction error → cortisol (Sapolsky 2004)
CORT_FAILURE_GAIN = 0.06      # Failure outcome → cortisol (McEwen 2007)
CORT_RELIEF_GAIN = 0.03       # Success → cortisol drop (Kirschbaum et al. 1995)

# EPI (Adrenal Medulla — Sympathetic NS)
EPI_AROUSAL_GAIN = 0.25       # Arousal → epinephrine (Cannon 1929)
EPI_THREAT_GAIN = 0.12        # Threat detection → epinephrine spike (Gold & Van Buskirk 1975)

# GABA (Cortical Interneurons — Inhibitory tone, Buzsaki 2006)
GABA_BASELINE = 0.5           # Tonic inhibitory tone
GABA_DECAY_RATE = 0.88        # Moderate-slow, ~6 cycles half-life
GABA_CONFLICT_GAIN = 0.12     # Conflict → inhibitory braking (Aron 2007)
GABA_EI_COMPENSATION = 0.10   # High excitation → compensatory inhibition (Isaacson & Scanziani 2011)
GABA_SUPPRESSION_FACTOR = 0.08  # High GABA → suppress excitatory NTs


class NeuromodulatorController:
    """Controls dynamic neuromodulator levels based on brain region signals.

    The pipeline calls specific methods at anatomically correct points.
    All phasic responses are additive deltas on top of current values.
    decay() should be called once per interaction cycle to drift toward baseline.
    """

    def __init__(self, neuromodulators: Neuromodulators):
        self.neuromodulators = neuromodulators
        self._last_arousal: float = 0.5  # Track arousal to scale processing boosts

    # ── VTA/SNc: Dopamine (Schultz 1997, Grace 2000) ────────────

    def on_prediction_error(
        self, error: float, predicted: str, actual: str,
    ) -> None:
        """Cerebellum prediction error → DA update.

        Positive RPE (better than expected) → phasic DA burst.
        Negative RPE (worse than expected) → phasic DA dip.
        """
        if predicted == "success" and actual == "failure":
            delta = -error * DA_GAIN
        elif predicted == "failure" and actual == "success":
            delta = (1.0 - error) * DA_GAIN
        else:
            delta = -error * DA_GAIN * 0.3 if error > 0.3 else error * DA_GAIN * 0.2

        # Apply via reward_signal alias (maps to dopamine internally)
        self.neuromodulators.reward_signal += delta

        # Large |RPE| also boosts ACh (surprise → learn)
        rpe_magnitude = abs(delta)
        if rpe_magnitude > 0.1:
            self.neuromodulators.learning_rate += rpe_magnitude * ACH_SURPRISE_GAIN

        # Large negative RPE → cortisol (uncontrollable stress, Sapolsky 2004)
        if error > 0.5:
            self.neuromodulators.cortisol += error * CORT_ERROR_GAIN

    # ── Locus Coeruleus: NE (Aston-Jones & Cohen 2005) ──────────

    def on_emotional_arousal(self, arousal: float) -> None:
        """Amygdala arousal → BIDIRECTIONAL NT modulation.

        High arousal (>0.5): activates NE, DA, CORT, EPI (fight-or-flight)
        Low arousal (<0.3): SUPPRESSES NE, EPI, DA (sleepy/calm state)
        Moderate (0.3-0.5): minimal change (baseline)

        This is bidirectional — the user's emotional state directly shapes
        the agent's neurochemical state. "졸리다" = low arousal = NE/EPI drop.

        References:
          - Aston-Jones & Cohen 2005: NE tracks arousal bidirectionally
          - Bromberg-Martin et al. 2010: DA salience/motivation
          - Cannon 1929: EPI sympathetic activation/deactivation
        """
        # ── Low arousal: actively pull NTs toward low state ──
        # Weight is strong enough to override processing boosts
        if arousal < 0.3:
            suppression = (0.3 - arousal) * 0.8  # max 0.24 at arousal=0
            self.neuromodulators.urgency -= suppression          # NE drops (sleepy)
            self.neuromodulators.epinephrine -= suppression       # EPI drops (relaxed)
            self.neuromodulators.dopamine -= suppression * 0.7    # DA drops (low motivation)
            self.neuromodulators.acetylcholine -= suppression * 0.5  # ACh drops (low learning drive)
            # Store arousal for other methods to scale their boosts
            self._last_arousal = arousal
            return  # No further boosts for low arousal

        self._last_arousal = arousal
        # ── Moderate to high arousal: boost systems ──
        # NE: arousal drives alertness, Yerkes-Dodson modulated
        yd_factor = self._yerkes_dodson(arousal, optimal=0.5)
        ne_delta = arousal * NE_AROUSAL_GAIN * yd_factor
        self.neuromodulators.urgency += ne_delta

        # DA: arousal drives salience/motivational engagement (Bromberg-Martin et al. 2010)
        # High arousal = highly salient stimulus → strong DA response
        # "총맞아 죽을뻔했어" (arousal~0.9) should produce visible DA spike
        if arousal > 0.3:
            da_delta = (arousal - 0.3) * 0.45  # Stronger: 0.3→0.45
            self.neuromodulators.dopamine += da_delta

        # Cortisol: HPA activates above moderate arousal threshold
        if arousal > 0.4:
            self.neuromodulators.cortisol += (arousal - 0.4) * CORT_STRESS_GAIN

        # Epinephrine: sympathetic response
        if arousal > 0.2:
            self.neuromodulators.epinephrine += (arousal - 0.2) * EPI_AROUSAL_GAIN
        # Threat detection → epinephrine spike
        if arousal > 0.7:
            self.neuromodulators.epinephrine += EPI_THREAT_GAIN

    def on_conflict(self, conflict_score: float) -> None:
        """ACC conflict detection → NE + ACh phasic burst + GABA braking.

        Conflict = uncertain situation → need alertness AND learning.
        GABA braking prevents impulsive responding (Aron 2007).
        """
        self.neuromodulators.urgency += conflict_score * NE_CONFLICT_GAIN
        self.neuromodulators.learning_rate += conflict_score * ACH_UNCERTAINTY_GAIN
        self.neuromodulators.gaba += conflict_score * GABA_CONFLICT_GAIN

    # ── Nucleus Basalis: ACh (Hasselmo 2006) ─────────────────────

    def on_novelty(self, novelty: float) -> None:
        """SN novelty assessment → ACh modulation.

        High novelty → high ACh → encode more strongly.
        Low novelty → low ACh → rely on existing memories.
        """
        delta = (novelty - 0.5) * ACH_NOVELTY_GAIN
        self.neuromodulators.learning_rate += delta

    # ── Dorsal Raphe: 5-HT (Doya 2002) ──────────────────────────

    def on_system_state(
        self, pending_requests: int, error_rate: float,
    ) -> None:
        """Hypothalamus system state → NE + 5-HT.

        High load → urgency up (additive delta, not absolute set).
        High error → patience down (additive delta).
        Previous implementation used absolute SET which overwrote
        all accumulated phasic NE/5-HT changes — now uses delta.
        """
        load_factor = min(1.0, pending_requests / 10.0)
        ne_delta = load_factor * NE_LOAD_GAIN
        self.neuromodulators.urgency += ne_delta

        # High error rate → reduce patience (delta from current, not from baseline)
        sht_delta = (0.5 - error_rate) * SEROTONIN_ERROR_GAIN * 0.3
        self.neuromodulators.patience += sht_delta

    def on_reward_outcome(self, success: bool) -> None:
        """Action outcome → 5-HT nudge + cortisol modulation + DA nudge.

        Scaled by current arousal — low arousal dampens processing boosts.
        Success: patience grows, cortisol drops, DA rises (Kirschbaum et al. 1995).
        Failure: patience drops, cortisol rises, DA dips (McEwen 2007 allostatic load).
        """
        # Scale processing boosts by arousal (low arousal = muted response)
        arousal_scale = max(0.2, min(1.0, self._last_arousal * 2))
        sht_delta = SEROTONIN_REWARD_GAIN if success else -SEROTONIN_REWARD_GAIN * 2
        self.neuromodulators.patience += sht_delta * arousal_scale

        # DA: success → small positive reinforcement signal, failure → dip
        da_delta = 0.025 if success else -0.04
        self.neuromodulators.dopamine += da_delta * arousal_scale

        if success:
            self.neuromodulators.cortisol -= CORT_RELIEF_GAIN
        else:
            self.neuromodulators.cortisol += CORT_FAILURE_GAIN

    # ── Decay (inter-event baseline drift) ───────────────────────

    def decay(self) -> None:
        """Drift all neuromodulators toward baseline.

        Called once per interaction cycle. Uses exponential decay:
          value = baseline + (value - baseline) * rate

        Different time constants per system:
          - DA, NE, ACh: DECAY_RATE (0.85) — half-life ~5 cycles
          - 5-HT: SEROTONIN_DECAY_RATE (0.90) — half-life ~7 cycles
          - Cortisol: CORT_DECAY_RATE (0.93) — half-life ~10 cycles
          - Epinephrine: EPI_DECAY_RATE (0.75) — fast, ~2.5 cycles
        """
        nm = self.neuromodulators
        nm.dopamine = DA_BASELINE + (nm.dopamine - DA_BASELINE) * DECAY_RATE
        nm.norepinephrine = NE_BASELINE + (nm.norepinephrine - NE_BASELINE) * DECAY_RATE
        nm.acetylcholine = ACH_BASELINE + (nm.acetylcholine - ACH_BASELINE) * DECAY_RATE
        nm.serotonin = SEROTONIN_BASELINE + (nm.serotonin - SEROTONIN_BASELINE) * SEROTONIN_DECAY_RATE
        nm.cortisol = CORT_BASELINE + (nm.cortisol - CORT_BASELINE) * CORT_DECAY_RATE
        nm.epinephrine = EPI_BASELINE + (nm.epinephrine - EPI_BASELINE) * EPI_DECAY_RATE
        nm.gaba = GABA_BASELINE + (nm.gaba - GABA_BASELINE) * GABA_DECAY_RATE
        # Inter-NT crosstalk
        self._apply_crosstalk()

    # ── Inter-NT Crosstalk ──────────────────────────────────────────

    def _apply_crosstalk(self) -> None:
        """Inter-NT interactions (crosstalk) — real neurochemical systems are not independent.

        References:
          - Porter & Bhatt (2008): CORT suppresses 5-HT transporter expression
          - Devoto & Flore (2006): DA-NE shared tyrosine hydroxylase pathway
          - Threlfell & Cragg (2011): ACh modulates striatal DA release
        """
        nm = self.neuromodulators

        # CORT↑ → 5-HT↓ (stress reduces patience)
        if nm.cortisol > 0.6:
            cort_excess = nm.cortisol - 0.6
            nm.serotonin -= cort_excess * 0.15  # Moderate suppression

        # DA-NE competition (shared precursor tyrosine)
        # When one is very high (>0.7), it slightly suppresses the other
        if nm.dopamine > 0.7:
            nm.norepinephrine -= (nm.dopamine - 0.7) * 0.1
        if nm.norepinephrine > 0.7:
            nm.dopamine -= (nm.norepinephrine - 0.7) * 0.1

        # ACh → DA modulation (cholinergic interneurons in striatum)
        # High ACh slightly boosts DA (learning state enhances reward sensitivity)
        if nm.acetylcholine > 0.6:
            nm.dopamine += (nm.acetylcholine - 0.6) * 0.08

        # High CORT → NE↑ (stress amplifies alertness via HPA→LC pathway)
        if nm.cortisol > 0.65:
            nm.norepinephrine += (nm.cortisol - 0.65) * 0.1

        # E/I homeostatic compensation (Isaacson & Scanziani 2011)
        excitation = (nm.norepinephrine + nm.dopamine) / 2.0
        if excitation > 0.65:
            nm.gaba += (excitation - 0.65) * GABA_EI_COMPENSATION
        # High GABA → suppress excitatory NTs (cortical inhibition)
        if nm.gaba > 0.7:
            gaba_excess = nm.gaba - 0.7
            nm.norepinephrine -= gaba_excess * GABA_SUPPRESSION_FACTOR
            nm.dopamine -= gaba_excess * GABA_SUPPRESSION_FACTOR * 0.5

    # ── Yerkes-Dodson Nonlinear Curve ───────────────────────────────

    def _yerkes_dodson(self, arousal: float, optimal: float = 0.5) -> float:
        """Inverted-U performance curve (Yerkes & Dodson 1908).

        Returns a multiplier [0.0, 1.0] where:
          - arousal near optimal → 1.0 (peak performance)
          - arousal too low or too high → reduced performance

        Formula: exp(-((arousal - optimal)^2) / (2 * 0.15^2))
        This gives a Gaussian centered at optimal with σ=0.15
        """
        import math
        sigma = 0.15
        return math.exp(-((arousal - optimal) ** 2) / (2 * sigma ** 2))
