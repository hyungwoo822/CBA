"""Neuromodulator state — neurotransmitter and hormone levels.

Maps abstract cognitive functions to their neurochemical substrates:
  - Dopamine (DA): reward prediction error, motivation (Schultz 1997, Grace 2000)
  - Norepinephrine (NE): arousal, alertness, urgency (Aston-Jones & Cohen 2005)
  - Serotonin (5-HT): patience, mood, temporal discounting (Doya 2002)
  - Acetylcholine (ACh): learning, novelty, encoding plasticity (Hasselmo 2006)
  - Cortisol (CORT): HPA axis stress response (Sapolsky 2004, McEwen 2007)
  - Epinephrine (EPI): sympathetic fight-or-flight (Cannon 1929, Cahill & McGaugh 1998)
  - GABA: inhibitory tone, E/I balance (Buzsaki 2006, Isaacson & Scanziani 2011)

All values on [0.0, 1.0] scale. 0.5 = tonic baseline (normal resting state).
"""
from __future__ import annotations


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class Neuromodulators:
    def __init__(self):
        # Primary neurotransmitters
        self._dopamine = 0.5         # DA — tonic firing baseline (Grace 2000)
        self._norepinephrine = 0.5   # NE — moderate alertness
        self._serotonin = 0.5        # 5-HT — moderate patience
        self._acetylcholine = 0.5    # ACh — moderate plasticity
        # Hormones
        self._cortisol = 0.5         # CORT — no stress
        self._epinephrine = 0.5      # EPI — no arousal
        # Inhibitory
        self._gaba = 0.5            # GABA — tonic inhibition (Buzsaki 2006)

    # ── Dopamine (DA) — VTA/SNc reward prediction error ──────────
    @property
    def dopamine(self) -> float:
        return self._dopamine

    @dopamine.setter
    def dopamine(self, v: float) -> None:
        self._dopamine = _clamp(v)

    # ── Norepinephrine (NE) — Locus Coeruleus arousal ───────────
    @property
    def norepinephrine(self) -> float:
        return self._norepinephrine

    @norepinephrine.setter
    def norepinephrine(self, v: float) -> None:
        self._norepinephrine = _clamp(v)

    # ── Serotonin (5-HT) — Dorsal Raphe patience/mood ───────────
    @property
    def serotonin(self) -> float:
        return self._serotonin

    @serotonin.setter
    def serotonin(self, v: float) -> None:
        self._serotonin = _clamp(v)

    # ── Acetylcholine (ACh) — Nucleus Basalis learning ───────────
    @property
    def acetylcholine(self) -> float:
        return self._acetylcholine

    @acetylcholine.setter
    def acetylcholine(self, v: float) -> None:
        self._acetylcholine = _clamp(v)

    # ── Cortisol (CORT) — HPA axis stress ────────────────────────
    @property
    def cortisol(self) -> float:
        return self._cortisol

    @cortisol.setter
    def cortisol(self, v: float) -> None:
        self._cortisol = _clamp(v)

    # ── Epinephrine (EPI) — Sympathetic fight-or-flight ──────────
    @property
    def epinephrine(self) -> float:
        return self._epinephrine

    @epinephrine.setter
    def epinephrine(self, v: float) -> None:
        self._epinephrine = _clamp(v)

    # ── GABA — Cortical inhibitory tone (Buzsaki 2006) ────────────
    @property
    def gaba(self) -> float:
        return self._gaba

    @gaba.setter
    def gaba(self, v: float) -> None:
        self._gaba = _clamp(v)

    # ── Backward-compatible aliases ──────────────────────────────
    # Existing code uses these functional names; they map 1:1 to NTs.

    @property
    def urgency(self) -> float:
        return self._norepinephrine

    @urgency.setter
    def urgency(self, v: float) -> None:
        self._norepinephrine = _clamp(v)

    @property
    def learning_rate(self) -> float:
        return self._acetylcholine

    @learning_rate.setter
    def learning_rate(self, v: float) -> None:
        self._acetylcholine = _clamp(v)

    @property
    def patience(self) -> float:
        return self._serotonin

    @patience.setter
    def patience(self, v: float) -> None:
        self._serotonin = _clamp(v)

    @property
    def inhibition(self) -> float:
        return self._gaba

    @inhibition.setter
    def inhibition(self, v: float) -> None:
        self._gaba = _clamp(v)

    @property
    def reward_signal(self) -> float:
        """Old [-1, 1] scale mapped from dopamine [0, 1]."""
        return (self._dopamine - 0.5) * 2.0

    @reward_signal.setter
    def reward_signal(self, v: float) -> None:
        """Accept old [-1, 1] scale, store as dopamine [0, 1]."""
        self._dopamine = _clamp(v / 2.0 + 0.5)

    # ── Utility ──────────────────────────────────────────────────

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def snapshot(self) -> dict:
        """Return all 7 neurotransmitter levels for dashboard."""
        return {
            "dopamine": self._dopamine,
            "norepinephrine": self._norepinephrine,
            "serotonin": self._serotonin,
            "acetylcholine": self._acetylcholine,
            "cortisol": self._cortisol,
            "epinephrine": self._epinephrine,
            "gaba": self._gaba,
        }

    def load_from(self, state: dict) -> None:
        """Restore neuromodulator state from persisted values.

        Used on startup to resume the brain's neurochemical state
        from the previous session. The brain doesn't reset between
        conversations — chronic stress keeps cortisol elevated,
        repeated rewards shift dopamine sensitivity (McEwen 2007).
        """
        self._dopamine = _clamp(state.get("dopamine", 0.5))
        self._norepinephrine = _clamp(state.get("norepinephrine", 0.5))
        self._serotonin = _clamp(state.get("serotonin", 0.5))
        self._acetylcholine = _clamp(state.get("acetylcholine", 0.5))
        self._cortisol = _clamp(state.get("cortisol", 0.5))
        self._epinephrine = _clamp(state.get("epinephrine", 0.5))
        self._gaba = _clamp(state.get("gaba", 0.5))
