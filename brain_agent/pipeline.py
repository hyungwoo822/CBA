"""Processing Pipeline — 7-Phase Neural Processing Model.

Restructured to match the academic 7-phase neural processing pipeline:

Phase 1 (Sensory Input):    Thalamus relay → V1/A1 (Sherman & Guillery 2006)
Phase 2 (Dual Streams):     Ventral("what") ∥ Dorsal("how") (Hickok & Poeppel 2007)
Phase 3 (Integration):      pSTS binding + Amygdala + Attention gating
Phase 6 (Retrieval):        Hippocampus + PFC + Angular Gyrus (Squire 2004)
  [Executive Processing]:   PFC → ACC → BasalGanglia → Cerebellum → Execute
Phase 4 (Encoding):         Hippocampus + Amygdala co-encoding (McGaugh 2004)
Phase 7 (Speech Production): Spt → Broca → M1 (Levelt 1989)
Phase 5 (Consolidation):    Periodic SWS check (Zielinski 2018)
"""
from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider
    from brain_agent.dashboard.emitter import DashboardEmitter

from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.activation_profile import compute_activation_profile, should_full_process
from brain_agent.core.neuromodulators import Neuromodulators
from brain_agent.core.neuromodulator_controller import NeuromodulatorController
from brain_agent.core.predictor import Predictor
from brain_agent.core.network_modes import TripleNetworkController, NetworkMode
from brain_agent.core.router import ThalamicRouter
from brain_agent.core.workspace import GlobalWorkspace
from brain_agent.memory.manager import MemoryManager
from brain_agent.memory.working_memory import WorkingMemoryItem
from brain_agent.regions.thalamus import Thalamus
from brain_agent.regions.amygdala import Amygdala
from brain_agent.regions.salience_network import SalienceNetworkRegion
from brain_agent.regions.prefrontal import PrefrontalCortex
from brain_agent.regions.acc import AnteriorCingulateCortex
from brain_agent.regions.basal_ganglia import BasalGanglia
from brain_agent.regions.cerebellum import Cerebellum, MINOR_ERROR_THRESHOLD
from brain_agent.regions.hypothalamus import Hypothalamus
from brain_agent.regions.visual_cortex import VisualCortex
from brain_agent.regions.auditory_cortex import AuditoryCortexLeft, AuditoryCortexRight
from brain_agent.regions.wernicke import WernickeArea
from brain_agent.regions.broca import BrocaArea
from brain_agent.regions.brainstem import Brainstem
from brain_agent.regions.vta import VentralTegmentalArea
from brain_agent.regions.corpus_callosum import CorpusCallosum
from brain_agent.regions.angular_gyrus import AngularGyrus
from brain_agent.regions.psts import PosteriorSuperiorTemporalSulcus
from brain_agent.regions.spt import SylvianParietalTemporal
from brain_agent.regions.motor_cortex import MotorCortex
from brain_agent.regions.mpfc import MedialPrefrontalCortex
from brain_agent.regions.insula import Insula
from brain_agent.regions.tpj import TemporoparietalJunction
from dataclasses import dataclass, field



MAX_REPROCESS = 2  # Maximum reprocessing iterations (Lamme 2006)
REPROCESS_CONFIDENCE_THRESHOLD = 0.4  # Below this → re-deliberate


def _classify_complexity(
    comprehension: dict, has_procedure: bool,
) -> str:
    """Classify input complexity for adaptive processing depth.

    Returns 'fast', 'standard', or 'full' based on Wernicke comprehension
    and procedural cache status.

    References:
      - Schneider & Shiffrin (1977): Automatic vs controlled processing
      - Posner & Snyder (1975): Two-process theory of attention
    """
    complexity = comprehension.get("complexity", "moderate")
    intent = comprehension.get("intent", "unknown")

    # Fast path: routine + procedural cache hit
    if has_procedure and complexity == "simple":
        return "fast"

    # Fast path: greetings, confirmations, single-word responses
    if intent in ("greeting", "confirmation", "farewell") and complexity == "simple":
        return "fast"

    # Full path: complex analytical / creative requests
    if complexity in ("complex", "very_complex"):
        return "full"

    return "standard"


@dataclass
class PipelineResult:
    response: str = ""
    actions_taken: list[dict] = field(default_factory=list)
    network_mode: str = ""
    signals_processed: int = 0
    memories_retrieved: list[dict] = field(default_factory=list)
    memory_encoded: bool = False
    from_cache: bool = False


class ProcessingPipeline:
    """Orchestrates the 7-phase neural processing pipeline."""

    def __init__(
        self,
        memory: MemoryManager,
        llm_provider: LLMProvider | None = None,
        emitter: DashboardEmitter | None = None,
        tool_registry=None,
        max_tool_iterations: int = 10,
    ):
        self.memory = memory
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations
        self._llm_provider = llm_provider  # Keep direct reference for Phase 5 consolidation
        self.neuromodulators = Neuromodulators()
        self.network_ctrl = TripleNetworkController()
        self.router = ThalamicRouter(
            network_ctrl=self.network_ctrl, neuromodulators=self.neuromodulators
        )
        self.workspace = GlobalWorkspace()

        # ── Phase 1: Sensory regions ──────────────────────────────────
        self.thalamus = Thalamus()                          # LGN/MGN relay
        self.visual_cortex = VisualCortex(llm_provider=llm_provider)  # V1
        self.auditory_cortex_l = AuditoryCortexLeft(llm_provider=llm_provider)  # A1 left (speech)
        self.auditory_cortex_r = AuditoryCortexRight()      # A1 right (prosody)

        # ── Phase 2: Dual-stream regions ──────────────────────────────
        self.wernicke = WernickeArea(llm_provider=llm_provider)  # Ventral auditory endpoint
        self.angular_gyrus = AngularGyrus()                 # Visual ventral integration
        self.spt = SylvianParietalTemporal()                # Dorsal auditory-motor

        # ── Phase 3: Integration + attention gating ───────────────────
        self.psts = PosteriorSuperiorTemporalSulcus()       # Multisensory binding
        self.amygdala = Amygdala(llm_provider=llm_provider) # Emotional evaluation
        self.salience = SalienceNetworkRegion(network_ctrl=self.network_ctrl)
        self.insula = Insula(neuromodulators=self.neuromodulators)  # Interoception (Craig 2009)

        # ── Phase 3: Identity regions (schema layer) ────────────────────
        self.mpfc = MedialPrefrontalCortex()             # Self-model (Northoff 2006)
        self.tpj = TemporoparietalJunction()             # User-model (Frith & Frith 2006)

        # ── Executive processing ──────────────────────────────────────
        self.pfc = PrefrontalCortex(llm_provider=llm_provider)
        self.acc = AnteriorCingulateCortex()
        self.basal_ganglia = BasalGanglia()
        self.cerebellum = Cerebellum()
        self.corpus_callosum = CorpusCallosum()

        # ── Phase 7: Speech production ────────────────────────────────
        self.broca = BrocaArea(llm_provider=llm_provider)   # Language formulation
        self.motor_cortex = MotorCortex()                   # Final output (M1)

        # ── Subcortical / homeostatic ─────────────────────────────────
        self.hypothalamus = Hypothalamus(neuromodulators=self.neuromodulators)
        self.neuro_ctrl = NeuromodulatorController(self.neuromodulators)
        self.vta = VentralTegmentalArea()
        self.brainstem_region = Brainstem()

        self._emitter = emitter
        self.predictor = Predictor()  # Predictive coding (Friston 2005)

        # Dreaming engine (Diekelmann & Born 2010) — recall-based promotion
        from brain_agent.memory.dreaming import DreamingEngine
        self.dreaming = DreamingEngine(tracker=self.memory.recall_tracker, mode="core")

        # Wire ACh getter for consolidation gating (Hasselmo 2006)
        self.memory.set_neuromodulators(lambda: self.neuromodulators.acetylcholine)
        # Wire cortisol getter for retrieval inhibition (de Quervain 2000)
        self.memory.set_cortisol_accessor(lambda: self.neuromodulators.cortisol)

    # ── Brain state persistence (McEwen 2007: allostatic load persists) ──

    def _all_regions(self) -> list:
        """Return all instantiated brain region objects."""
        return [
            self.thalamus, self.visual_cortex, self.auditory_cortex_l,
            self.auditory_cortex_r, self.wernicke, self.angular_gyrus,
            self.spt, self.psts, self.amygdala, self.salience,
            self.insula, self.mpfc, self.tpj,
            self.pfc, self.acc, self.basal_ganglia, self.cerebellum,
            self.corpus_callosum, self.broca, self.motor_cortex,
            self.hypothalamus, self.vta, self.brainstem_region,
        ]

    async def restore_brain_state(self) -> None:
        """Load persisted neuromodulator and activation state from DB.

        Called on startup to resume the brain's state from previous session.
        Models inter-session decay — NOT a hard reset.

        The brain doesn't reset to zero between conversations. During sleep:
          - Cortisol: barely decays (slow HPA, McEwen 2007) → 95% carry-over
          - 5-HT (mood/patience): moderate carry-over → 70%
          - DA, NE, ACh: faster decay → 50% carry-over
          - EPI: fastest decay → 20% carry-over (transient by nature)

        This means chronic stress (high cortisol) persists across sessions,
        mood (5-HT) carries over partially, and acute states (NE, EPI) fade.
        """
        store = self.memory.brain_state
        nt_state = await store.load_neuromodulators()

        # Inter-session decay: each NT drifts toward baseline (0.5) by its rate
        def _carry(saved: float, rate: float) -> float:
            """rate=1.0 = full carry, rate=0.0 = full reset to 0.5"""
            return 0.5 + (saved - 0.5) * rate

        self.neuromodulators.load_from({
            "dopamine": _carry(nt_state.get("dopamine", 0.5), 0.5),
            "norepinephrine": _carry(nt_state.get("norepinephrine", 0.5), 0.5),
            "serotonin": _carry(nt_state.get("serotonin", 0.5), 0.7),
            "acetylcholine": _carry(nt_state.get("acetylcholine", 0.5), 0.5),
            "cortisol": _carry(nt_state.get("cortisol", 0.5), 0.95),
            "epinephrine": _carry(nt_state.get("epinephrine", 0.5), 0.2),
            "gaba": _carry(nt_state.get("gaba", 0.5), 0.6),
        })
        # Restore region activations
        activations = await store.load_region_activations()
        for region in self._all_regions():
            if region.name in activations:
                region.activation_level = activations[region.name]

        # Load identity facts into mPFC and TPJ
        try:
            identity = await self.memory.retrieve_identity()
            self.mpfc.update_from_graph_facts(identity.get("self_model", []))
            self.tpj.update_from_graph_facts(identity.get("user_model", []))
        except Exception:
            pass  # Identity load is best-effort

    async def save_brain_state(self) -> None:
        """Persist current brain state to DB after request processing."""
        store = self.memory.brain_state
        # Save neuromodulators
        await store.save_neuromodulators(self.neuromodulators.snapshot())
        # Save region activations
        activations = {r.name: r.activation_level for r in self._all_regions()}
        await store.save_region_activations(activations)

    # ── Utility methods ───────────────────────────────────────────────

    def _route(self, signal: Signal) -> list[str]:
        return self.router.resolve_targets(signal)

    def _is_active(self, region_name: str) -> bool:
        """Check if region should process in current network mode (Menon 2011)."""
        from brain_agent.core.network_modes import ALWAYS_ACTIVE
        if region_name in ALWAYS_ACTIVE:
            return True
        return self.network_ctrl.is_region_active(region_name)

    async def _emit(self, method: str, *args, **kwargs) -> None:
        if self._emitter:
            fn = getattr(self._emitter, method, None)
            if fn:
                await fn(*args, **kwargs)

    @staticmethod
    def _strip_korean_particles(text: str) -> str:
        """Strip trailing Korean particles from entity names."""
        import re
        # Order matters: longer particles first
        particles = [
            '이야', '에서', '한테', '야', '는', '은', '가', '이',
            '를', '을', '도', '에', '의', '로', '으로',
        ]
        for p in particles:
            if text.endswith(p) and len(text) > len(p):
                return text[:-len(p)]
        return text

    async def _store_identity_facts_realtime(self, relations: list, source: str = "realtime") -> int:
        """Store durable user-related facts as identity_facts immediately.

        Stores ALL relations that involve the user's world — not just
        'user → X' triples, but also multi-entity edges like
        'grandmother → visit → user' or 'coffee → increase → dopamine'.

        This enables a richer agent user model that mirrors the user's
        actual knowledge graph structure.

        Returns the number of facts stored.
        """
        DURABLE_CATEGORIES = {"IDENTITY", "PREFERENCE", "ATTRIBUTE", "SOCIAL",
                              "ACTION", "EMOTION", "CAUSAL", "SPATIAL"}
        stored = 0
        for rel in relations:
            if len(rel) < 3:
                continue
            subj, relation, obj = str(rel[0]).lower(), str(rel[1]), str(rel[2])
            # Strip Korean particles from entity values
            obj = self._strip_korean_particles(obj)
            subj = self._strip_korean_particles(subj)
            try:
                conf = float(rel[3]) if len(rel) >= 4 else 0.7
            except (ValueError, TypeError):
                conf = 0.7
            cat = str(rel[4]).upper() if len(rel) >= 5 else "GENERAL"

            if cat not in DURABLE_CATEGORIES or conf < 0.6:
                continue

            # Generate semantic key — preserve multi-entity structure
            if subj == "user":
                # User-centric: user → relation → object
                if cat == "IDENTITY":
                    key = relation.replace(" ", "_").lower()
                    value = obj
                else:
                    key = f"{cat.lower()}:{relation.replace(' ', '_')}:{obj.replace(' ', '_').lower()}"
                    value = f"{relation} {obj}"
            else:
                # Multi-entity: subject → relation → object (e.g., grandmother → visit → user)
                key = f"{cat.lower()}:{subj.replace(' ', '_')}:{relation.replace(' ', '_')}:{obj.replace(' ', '_').lower()}"
                value = f"{subj} {relation} {obj}"

            try:
                await self.memory.semantic.add_identity_fact(
                    "user_model", key, value, source=source, confidence=conf,
                )
                stored += 1
            except Exception:
                logger.debug("Failed to store identity fact: %s=%s", key, value, exc_info=True)
        if stored:
            logger.info("Realtime identity facts stored: %d (source=%s)", stored, source)
        return stored

    async def _extract_user_facts(self, user_input: str, comprehension: dict) -> dict | None:
        """Extract facts about the user directly from their input (not from PFC response).

        This captures what the user explicitly said — preferences, states, habits —
        without the agent's interpretation layer. Produces richer user-centric
        knowledge graph entries.
        """
        intent = comprehension.get("intent", "unknown")
        keywords = comprehension.get("keywords", [])
        prompt = (
            "Extract facts about the user from their input. Build a MULTI-ENTITY GRAPH:\n"
            "1. User-centric facts (user as subject: habits, preferences, states)\n"
            "2. User's world entities (people, places, objects connected to user's life)\n"
            "3. Causal/relational chains between entities\n\n"
            f"User said: {user_input}\n"
            f"Detected intent: {intent}\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            "Output ONLY valid JSON:\n"
            '{"entities": ["entity1", "entity2"], '
            '"relations": [["subject", "relation", "object", confidence, "CATEGORY"]]}\n\n'
            "Rules:\n"
            "- ALL entities MUST be English lowercase nouns. NEVER use Korean/non-English.\n"
            "- ALL relations MUST be English verb infinitives: 'like', 'eat', 'cause'\n"
            "- Build GRAPH STRUCTURE: multiple entities with edges between them.\n"
            "  NOT just 'user → X', but also 'X → relation → Y' edges.\n"
            "- Example: '할머니가 병원 오셨어' →\n"
            '  [["user","have","grandmother",1.0,"SOCIAL"],'
            '["grandmother","visit","user",1.0,"ACTION"],'
            '["grandmother","location","hospital",0.8,"SPATIAL"]]\n'
            "- Example: '119 전화했는데 기분 나빠' →\n"
            '  [["user","call","119",1.0,"ACTION"],'
            '["119","cause","bad mood",0.8,"CAUSAL"],'
            '["user","experience","bad mood",1.0,"EMOTION"]]\n'
            "- Confidence: 1.0=explicit, 0.8=implied, 0.6=inferred, 0.4=guess\n"
            "- STRIP Korean particles from names: '형푸야'→'hyungpu', '현우는'→'hyunwoo'.\n"
            "  Particles: 야/이야/는/은/가/이/를/을/도/에/에서/한테/의\n"
            "  '나는 X야' → name is X, NOT Xya.\n"
            "- If user corrects a fact, output the CORRECTED value with confidence 1.0.\n"
            "- category: PREFERENCE|ACTION|ATTRIBUTE|SOCIAL|CAUSAL|SPATIAL|TEMPORAL|IDENTITY|EMOTION\n"
            "- Produce at least 2-3 triples per statement. Return empty ONLY if truly nothing."
        )
        try:
            response = await self.pfc.llm_provider.chat([
                {"role": "system", "content": "You extract structured facts from user messages. Return only JSON."},
                {"role": "user", "content": prompt},
            ], max_tokens=300, temperature=0.1)
            if response and response.content:
                import json as _json
                text = response.content.strip()
                if text.startswith("```"):
                    lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                data = _json.loads(text)
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.debug("User fact extraction LLM failed", exc_info=True)
        return None

    def _apply_gain(self, region, profile: dict[str, float]) -> None:
        """Scale a region's activation_level by its activation profile gain.

        This makes the dashboard show content-dependent activation patterns
        (Pessoa 2008). The gain is a multiplier: regions with low gain
        for the current input will show reduced activation.
        """
        gain = profile.get(region.name, 0.5)
        region.emit_activation(region.activation_level * gain)

    async def _step(self, duration: float = 0.03) -> None:
        if self._emitter:
            await asyncio.sleep(duration)

    async def _emit_region_io(self, region_name: str, signal_before: Signal, signal_after: Signal | None, processing: str) -> None:
        input_summary: dict = {
            "type": signal_before.type.value if hasattr(signal_before.type, 'value') else str(signal_before.type),
            "text": str(signal_before.payload.get("text", ""))[:80],
        }
        if signal_before.emotional_tag:
            input_summary["emotion"] = {"v": round(signal_before.emotional_tag.valence, 2), "a": round(signal_before.emotional_tag.arousal, 2)}

        output_summary: dict = {}
        if signal_after:
            output_summary["type"] = signal_after.type.value if hasattr(signal_after.type, 'value') else str(signal_after.type)
            for key in ("comprehension", "visual_features", "prosody", "input_type", "arousal_state", "conflict_score", "go_score", "predicted_outcome"):
                if key in signal_after.payload:
                    val = signal_after.payload[key]
                    output_summary[key] = val if not isinstance(val, (dict, list)) else val

        await self._emit("region_io", region_name, input_summary, output_summary, processing)

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================

    async def process_request(self, text: str = "", image: bytes | None = None, audio: bytes | None = None) -> PipelineResult:
        """Process a single user request through the 7-phase neural pipeline."""
        result = PipelineResult()
        signals_count = 0

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Sensory Input (Sherman & Guillery 2006)
        # Thalamus relays → V1/A1 sensory cortices
        # ══════════════════════════════════════════════════════════════

        # Determine input modality — this affects processing throughout the pipeline
        # (Hickok & Poeppel 2007: visual and auditory use different dual streams)
        if image:
            input_modality = "visual"
            initial_type = SignalType.IMAGE_INPUT
        elif audio:
            input_modality = "auditory"
            initial_type = SignalType.AUDIO_INPUT
        else:
            input_modality = "text"
            initial_type = SignalType.TEXT_INPUT

        input_signal = Signal(
            type=initial_type,
            source="user",
            payload={"text": text, "input_modality": input_modality},
        )

        # ── 1a. Thalamus: sensory relay (LGN for visual, MGN for auditory)
        # Text input also passes through thalamus — reading activates
        # visual processing (LGN) for orthographic analysis (Dehaene 2009)
        thalamus_before = input_signal
        # Thalamic attention gating (McAlonan et al. 2008)
        input_signal = await self.thalamus.process_with_attention(
            input_signal,
            goal_embedding=None,
            current_arousal=self.neuromodulators.epinephrine,
        )
        signals_count += 1
        await self._emit("region_activation", "thalamus", self.thalamus.activation_level, "active")
        await self._emit("signal_flow", "_input", "thalamus", input_signal.type.value, 0.6)
        await self._emit_region_io("thalamus", thalamus_before, input_signal, "sensory_relay")
        await self._emit("region_processing", "thalamus", "phase_1", f"Sensory relay: {input_modality} input, {len(text.split())} words")
        await self._step()

        # ── 1b. Route to sensory cortices based on modality
        # Each modality activates its specific cortical pathway
        if input_modality == "visual":
            input_signal.payload["image_data"] = image
            input_signal.type = SignalType.IMAGE_INPUT
            pre_visual = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
            input_signal = await self.visual_cortex.process(input_signal)
            signals_count += 1
            await self._emit_region_io("visual_cortex", pre_visual, input_signal, "V1_feature_extraction")
            await self._emit("region_activation", "visual_cortex", self.visual_cortex.activation_level, "active")
            await self._emit("signal_flow", "thalamus", "visual_cortex", "IMAGE_INPUT", 0.7)
            await self._step()
            if not text and input_signal.payload.get("visual_features", {}).get("description"):
                text = input_signal.payload["visual_features"]["description"]
                input_signal.payload["text"] = text
            # Keep IMAGE_INPUT type — downstream knows this is visual

        elif input_modality == "auditory":
            input_signal.payload["audio_data"] = audio
            input_signal.type = SignalType.AUDIO_INPUT
            pre_audio_l = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
            input_signal = await self.auditory_cortex_l.process(input_signal)
            signals_count += 1
            await self._emit_region_io("auditory_cortex_l", pre_audio_l, input_signal, "A1_speech_processing")
            await self._emit("region_activation", "auditory_cortex_l", self.auditory_cortex_l.activation_level, "active")
            await self._emit("signal_flow", "thalamus", "auditory_cortex_l", "AUDIO_INPUT", 0.7)
            await self._step()

            pre_audio_r = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
            input_signal = await self.auditory_cortex_r.process(input_signal)
            signals_count += 1
            await self._emit_region_io("auditory_cortex_r", pre_audio_r, input_signal, "A1_prosody_analysis")
            await self._emit("region_activation", "auditory_cortex_r", self.auditory_cortex_r.activation_level, "active")
            await self._emit("signal_flow", "thalamus", "auditory_cortex_r", "AUDIO_INPUT", 0.6)
            await self._step()

            text = input_signal.payload.get("text", text)
            # Keep AUDIO_INPUT type — downstream knows this is auditory

        else:
            # Text input: language is processed via the "reading pathway"
            # (Dehaene 2009: visual word form area → ventral stream)
            # Text goes directly to language processing (Phase 2)
            # but we still emit signal flow for dashboard visualization
            await self._emit("signal_flow", "thalamus", "wernicke", "TEXT_INPUT", 0.6)

        # ── 1c. Sensory buffer: register input (per-request cycle)
        self.memory.sensory.new_cycle()
        self.memory.sensory.register(input_signal.payload, modality=input_modality)

        self._route(input_signal)

        # Predictive coding: compute surprise (Friston 2005)
        prediction_surprise = 0.5  # Default neutral
        try:
            input_embedding = self.memory._embed_fn(text)
            prediction_surprise = self.predictor.compute_surprise(input_embedding)
            input_signal.metadata["prediction_surprise"] = prediction_surprise
        except Exception:
            pass  # Prediction is best-effort

        # ══════════════════════════════════════════════════════════════
        # Phase 2+3: Dual-Stream + Integration
        # Modality-specific dual streams (Hickok & Poeppel 2007):
        #
        # TEXT:  Wernicke (ventral="what") ∥ Amygdala (limbic)
        #        → pSTS merge (Beauchamp 2004)
        #
        # AUDIO: A1→Wernicke (ventral="what": STG→STS→MTG semantic)
        #        A1→Spt (dorsal="how": Spt→IFG articulatory mapping)
        #        ∥ Amygdala (limbic: emotional prosody evaluation)
        #        → pSTS merge
        #
        # IMAGE: V1→Angular Gyrus (ventral="what": V1→V2→V4→IT object recognition)
        #        V1→Spt (dorsal="where/how": spatial + action guidance)
        #        ∥ Amygdala (limbic: threat/reward visual evaluation)
        #        → pSTS merge
        # ══════════════════════════════════════════════════════════════

        # ── 2a. Parallel processing: Wernicke ∥ Amygdala ──────────
        # In the brain, ventral stream comprehension and limbic emotional
        # evaluation happen simultaneously on the same input.

        # Create independent signal copies for parallel processing
        wernicke_signal = Signal(
            type=input_signal.type, source=input_signal.source,
            payload=dict(input_signal.payload),
        )
        amygdala_signal = Signal(
            type=input_signal.type, source=input_signal.source,
            payload=dict(input_signal.payload),
        )

        # Run Wernicke and Amygdala in parallel when both need processing
        has_text = bool(wernicke_signal.payload.get("text"))
        if has_text:
            wernicke_result, amygdala_result = await asyncio.gather(
                self.wernicke.process(wernicke_signal),
                self.amygdala.process(amygdala_signal),
            )
        else:
            wernicke_result = wernicke_signal
            amygdala_result = await self.amygdala.process(amygdala_signal)
        signals_count += 2

        # Merge Wernicke comprehension back into main signal
        comprehension = {}
        if "comprehension" in wernicke_result.payload:
            input_signal.payload["comprehension"] = wernicke_result.payload["comprehension"]
            comprehension = input_signal.payload["comprehension"]

        # Always emit Wernicke processing status
        if comprehension:
            await self._emit("region_processing", "wernicke", "phase_2",
                f"Language comprehension: intent={comprehension.get('intent', '?')}, lang={comprehension.get('language', '?')}",
                {"keywords": comprehension.get("keywords", []), "complexity": comprehension.get("complexity", "?")})
        else:
            await self._emit("region_processing", "wernicke", "phase_2",
                f"Structural parse: {len(text.split())} words (no LLM)")

        # Modality-specific signal flow for dashboard visualization
        if input_modality == "auditory":
            # Auditory ventral: A1→STG→STS→MTG→Wernicke (semantic)
            await self._emit("signal_flow", "auditory_cortex_l", "wernicke", "AUDIO_INPUT", 0.7)
        elif input_modality == "visual":
            # Visual: V1 features fed to Wernicke for description comprehension
            await self._emit("signal_flow", "visual_cortex", "wernicke", "IMAGE_INPUT", 0.6)
        else:
            # Text: direct language pathway to Wernicke
            await self._emit("signal_flow", "thalamus", "wernicke", "TEXT_INPUT", 0.6)
        await self._emit("region_activation", "wernicke", self.wernicke.activation_level, "active")

        # Merge Amygdala emotional tag back into main signal
        input_signal.emotional_tag = amygdala_result.emotional_tag
        for key in ("amygdala_right", "amygdala_left", "amygdala_blend"):
            if key in amygdala_result.metadata:
                input_signal.metadata[key] = amygdala_result.metadata[key]
        await self._emit("region_activation", "amygdala", self.amygdala.activation_level, "active")
        await self._emit("signal_flow", "thalamus", "amygdala", input_signal.type.value, self.amygdala.activation_level or 0.3)
        # Always emit Amygdala processing status
        etag_for_emit = input_signal.emotional_tag
        amg_left = input_signal.metadata.get("amygdala_left", {})
        if etag_for_emit:
            await self._emit("region_processing", "amygdala", "phase_3",
                f"Emotional evaluation: {amg_left.get('primary_emotion', 'neutral')}, valence={etag_for_emit.valence:.2f}, arousal={etag_for_emit.arousal:.2f}",
                {"threat_level": amg_left.get("threat_level", "none")})
        else:
            await self._emit("region_processing", "amygdala", "phase_3",
                "Emotional baseline: neutral (no LLM)")
        await self._step()

        # ── Content-driven activation profiling (Pessoa 2008) ──
        activation_profile = compute_activation_profile(
            comprehension=comprehension,
            emotional_tag={"valence": input_signal.emotional_tag.valence,
                          "arousal": input_signal.emotional_tag.arousal}
                         if input_signal.emotional_tag else None,
            has_procedure=False,  # Not yet known; updated after Phase 6
        )
        input_signal.metadata["activation_profile"] = activation_profile
        await self._emit("region_processing", "pipeline", "activation_profile",
            f"Dynamic activation: {sum(1 for g in activation_profile.values() if g >= 0.5)}/{len(activation_profile)} regions active")

        # Retroactively scale already-processed regions by profile gain
        self._apply_gain(self.amygdala, activation_profile)
        self._apply_gain(self.wernicke, activation_profile)
        self._apply_gain(self.thalamus, activation_profile)

        # ── 2b. Ventral stream: modality-specific "what" processing
        # Visual: Angular Gyrus for cross-modal binding (V1→V2→V4→IT pathway)
        # Audio: Already processed via Wernicke (STG→STS→MTG pathway)
        # Text: Already processed via Wernicke
        if input_modality == "visual":
            angular_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
            input_signal = await self.angular_gyrus.process(input_signal)
            signals_count += 1
            await self._emit_region_io("angular_gyrus", angular_before, input_signal, "visual_ventral_integration")
            await self._emit("region_activation", "angular_gyrus", self.angular_gyrus.activation_level, "active")
            await self._emit("signal_flow", "visual_cortex", "angular_gyrus", "IMAGE_INPUT", 0.7)
            await self._step()
        elif input_signal.payload.get("prosody"):
            # Audio with prosody also triggers cross-modal binding
            angular_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
            input_signal = await self.angular_gyrus.process(input_signal)
            signals_count += 1
            await self._emit_region_io("angular_gyrus", angular_before, input_signal, "audio_semantic_binding")
            await self._emit("region_activation", "angular_gyrus", self.angular_gyrus.activation_level, "active")
            await self._emit("signal_flow", "auditory_cortex_r", "angular_gyrus", "AUDIO_INPUT", 0.6)
            await self._step()

        # ── 2c. Dorsal stream: modality-specific "how/where" processing
        # Audio dorsal: Spt→IFG articulatory mapping (Hickok & Poeppel 2007)
        # Visual dorsal: spatial location + action guidance (Goodale & Milner 1992)
        # Text: Spt maps orthographic → phonological (inner speech, Hickok 2012)
        spt_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
        input_signal = await self.spt.process(input_signal)
        signals_count += 1
        if input_modality == "auditory":
            processing_label = "dorsal_auditory_motor_mapping"
            flow_source = "auditory_cortex_l"
        elif input_modality == "visual":
            processing_label = "dorsal_visual_spatial_mapping"
            flow_source = "visual_cortex"
        else:
            processing_label = "dorsal_phonological_mapping"
            flow_source = "wernicke"
        await self._emit_region_io("spt", spt_before, input_signal, processing_label)
        await self._emit("region_activation", "spt", self.spt.activation_level, "active")
        await self._emit("signal_flow", flow_source, "spt", input_signal.type.value, 0.6)
        await self._step()

        # ── 3a. pSTS: merge all streams (Beauchamp 2004)
        psts_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
        input_signal = self.psts.integrate(input_signal)
        signals_count += 1
        await self._emit_region_io("psts", psts_before, input_signal, "multisensory_binding")
        await self._emit("region_activation", "psts", self.psts.activation_level, "active")
        await self._step()

        # Amygdala arousal drives neuromodulator response
        # No fake baseline — if amygdala returns 0.0, that means calm input
        arousal = (
            input_signal.emotional_tag.arousal if input_signal.emotional_tag
            else 0.0
        )
        self.neuro_ctrl.on_emotional_arousal(arousal)
        await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

        # ── 3c. Salience Network: evaluate novelty → mode switch
        prev_mode = self.network_ctrl.current_mode.value
        sn_before = input_signal
        await self.salience.process(input_signal)
        signals_count += 1
        new_mode = self.network_ctrl.current_mode.value
        if prev_mode != new_mode:
            await self._emit("network_switch", prev_mode, new_mode, "salience_evaluation")
        await self._emit("region_activation", "salience_network", self.salience.activation_level, "active")
        await self._emit("signal_flow", "amygdala", "salience_network", input_signal.type.value, 0.5)
        await self._emit_region_io("salience_network", sn_before, input_signal, "mode_evaluation")
        await self._step()

        # Neuromodulator: novelty → ACh (Hasselmo 2006)
        novelty = input_signal.metadata.get("computed_novelty", 0.5)
        # Predictive coding enhances novelty (Friston 2005)
        prediction_surprise = input_signal.metadata.get("prediction_surprise", 0.5)
        effective_novelty = max(novelty, prediction_surprise)
        self.neuro_ctrl.on_novelty(effective_novelty)
        await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

        # ── Insula: interoceptive monitoring (Craig 2009) ──
        insula_before = Signal(type=input_signal.type, source=input_signal.source, payload=dict(input_signal.payload))
        insula_before.emotional_tag = input_signal.emotional_tag
        input_signal = await self.insula.process(input_signal)
        signals_count += 1
        self._apply_gain(self.insula, activation_profile)
        await self._emit("region_activation", "insula", self.insula.activation_level, "active")
        await self._emit("signal_flow", "amygdala", "insula", "EMOTIONAL_TAG", 0.5)
        intero = input_signal.metadata.get("interoceptive_state", {})
        await self._emit("region_processing", "insula", "phase_3",
            f"Interoception: stress={intero.get('stress_level', 0):.2f}, energy={intero.get('energy_level', 0):.2f}")
        await self._step()

        # ── 3f. mPFC: self-referential processing (Northoff 2006) ──
        # Load identity facts from semantic store each request
        try:
            identity = await self.memory.retrieve_identity()
            self.mpfc.update_from_graph_facts(identity.get("self_model", []))
            self.tpj.update_from_graph_facts(identity.get("user_model", []))
        except Exception:
            pass
        await self.mpfc.process(input_signal)
        signals_count += 1
        self._apply_gain(self.mpfc, activation_profile)
        await self._emit("region_activation", "medial_pfc", self.mpfc.activation_level, "active")
        await self._emit("region_processing", "medial_pfc", "phase_3", "Self-model activated")
        await self._step()

        # ── 3g. TPJ: Theory of Mind — user modeling (Frith & Frith 2006) ──
        await self.tpj.process(input_signal)
        signals_count += 1
        self._apply_gain(self.tpj, activation_profile)
        await self._emit("region_activation", "tpj", self.tpj.activation_level, "active")
        await self._emit("signal_flow", "medial_pfc", "tpj", input_signal.type.value, 0.5)
        await self._emit("region_processing", "tpj", "phase_3", "User-model activated")
        await self._step()

        # ── 3d. Ensure ECN for task engagement
        if (
            input_signal.type in (SignalType.EXTERNAL_INPUT, SignalType.TEXT_INPUT, SignalType.IMAGE_INPUT, SignalType.AUDIO_INPUT)
            and self.network_ctrl.current_mode == NetworkMode.DMN
        ):
            self.network_ctrl.switch_to(NetworkMode.ECN, trigger="task_engagement")
            await self._emit("network_switch", "DMN", "ECN", "task_engagement")

        # ── 3e. Working Memory: load processed representation (Cowan 4±1)
        wm_meta = {
            "intent": comprehension.get("intent", "statement"),
            "keywords": comprehension.get("keywords", []),
            "complexity": comprehension.get("complexity", "simple"),
            "input_type": input_signal.payload.get("input_type", "unknown"),
            "arousal": input_signal.emotional_tag.arousal if input_signal.emotional_tag else 0.0,
        }
        wm_item = WorkingMemoryItem(content=text, slot="phonological", metadata=wm_meta)
        self.memory.working.load(wm_item)

        # ══════════════════════════════════════════════════════════════
        # Phase 6: Retrieval (Squire 2004, Tulving 2002)
        # Hippocampus + PFC + Angular Gyrus pattern completion
        # ══════════════════════════════════════════════════════════════

        wm_context = self.memory.working.get_context()
        retrieved = await self.memory.retrieve(
            query=text,
            context=wm_context,
            top_k=5,
        )
        result.memories_retrieved = retrieved

        # Attention-weighted retrieval boost
        attention_weight = input_signal.metadata.get("attention_weight", 0.5)
        if attention_weight > 0.6:
            for mem in retrieved:
                mem["score"] = mem.get("score", 0) * (1.0 + (attention_weight - 0.5) * 0.3)

        # Episodic buffer integration (Baddeley 2000)
        self.memory.working.bind_to_episodic_buffer(retrieved)

        # ── Sensory cortex reactivation (Wheeler et al. 2000)
        # Memory retrieval reactivates modality-specific sensory cortices:
        # visual memories → V1, auditory memories → A1.
        # This models the neural reinstatement effect during recall.
        for mem in retrieved:
            source = mem.get("source", "")
            entities = mem.get("entities", {})
            modality = entities.get("modality") if isinstance(entities, dict) else None
            if modality == "visual" or "visual" in source:
                self.visual_cortex.emit_activation(
                    min(1.0, self.visual_cortex.activation_level + 0.2)
                )
                await self._emit("region_activation", "visual_cortex",
                                 self.visual_cortex.activation_level, "memory_reactivation")
            elif modality == "auditory" or "audio" in source:
                self.auditory_cortex_l.emit_activation(
                    min(1.0, self.auditory_cortex_l.activation_level + 0.2)
                )
                await self._emit("region_activation", "auditory_cortex_l",
                                 self.auditory_cortex_l.activation_level, "memory_reactivation")

        await self._emit("signal_flow", "salience_network", "hippocampus", "RETRIEVE", 0.6)
        await self._emit("region_processing", "hippocampus", "phase_6",
            f"Retrieved {len(retrieved)} memories",
            {"top_score": round(retrieved[0].get('score', 0), 2) if retrieved else 0})
        await self._step()

        # Procedural cache check (Collins & Loftus 1975)
        cached_procedure = await self.memory.procedural.match(text)

        # Update activation profile with procedural info
        if cached_procedure:
            activation_profile = compute_activation_profile(
                comprehension=comprehension,
                emotional_tag={"valence": input_signal.emotional_tag.valence,
                              "arousal": input_signal.emotional_tag.arousal}
                             if input_signal.emotional_tag else None,
                has_procedure=True,
            )
            input_signal.metadata["activation_profile"] = activation_profile

        # ── Build upstream context for PFC (Miller & Cohen 2001) ──
        # Accumulate ALL upstream processing results so PFC has full context
        input_signal.metadata["retrieved_memories"] = retrieved
        if cached_procedure:
            input_signal.metadata["cached_procedure"] = cached_procedure
        input_signal.metadata["network_mode"] = self.network_ctrl.current_mode.value

        # Upstream context: everything PFC needs from Phases 1-3 and 6
        input_signal.metadata["upstream_context"] = {
            # Phase 1: input modality
            "input_modality": input_modality,
            # Phase 2: Wernicke comprehension (ventral stream)
            "comprehension": comprehension,
            # Phase 3: Amygdala emotional evaluation (limbic)
            "amygdala_left": input_signal.metadata.get("amygdala_left", {}),
            "amygdala_right": input_signal.metadata.get("amygdala_right", {}),
            # Phase 3: Neuromodulator state (6-NT system)
            "neuromodulators": self.neuromodulators.snapshot(),
            # Phase 3: Identity context from mPFC and TPJ
            "self_context": input_signal.metadata.get("self_context", ""),
            "user_context": input_signal.metadata.get("user_context", ""),
            # Phase 3: Insula interoceptive state (Craig 2009)
            "interoceptive_state": input_signal.metadata.get("interoceptive_state", {}),
        }

        await self._emit("signal_flow", "hippocampus", "prefrontal_cortex", "RETRIEVE", 0.7)

        # ── Adaptive processing depth (Schneider & Shiffrin 1977) ──
        processing_depth = _classify_complexity(comprehension, has_procedure=bool(cached_procedure))
        await self._emit("region_processing", "pipeline", "routing",
            f"Processing depth: {processing_depth}")

        # ══════════════════════════════════════════════════════════════
        # Executive Processing with Recurrent Loop (Lamme 2006)
        # PFC → ACC → [confidence check] → re-PFC if needed
        # ══════════════════════════════════════════════════════════════

        plan_signal = None
        conflict = None
        for reprocess_iter in range(MAX_REPROCESS + 1):
            # ── PFC: plan with memory context (LLM reasoning)
            pfc_before = input_signal
            plan_signal = await self.pfc.process(input_signal)
            signals_count += 1
            self._apply_gain(self.pfc, activation_profile)
            await self._emit("region_activation", "prefrontal_cortex", self.pfc.activation_level, "high_activity")
            await self._emit("signal_flow", "salience_network", "prefrontal_cortex", input_signal.type.value, 0.8)
            await self._emit_region_io("prefrontal_cortex", pfc_before, plan_signal, "planning")
            if plan_signal:
                # Extract PFC response for preview (may not be in result.response yet)
                pfc_text = ""
                for act in plan_signal.payload.get("actions", []):
                    t = act.get("args", {}).get("text", "")
                    if t:
                        pfc_text = t
                        break
                preview = pfc_text[:120] if pfc_text else "(processing...)"
                await self._emit("region_processing", "prefrontal_cortex", "executive",
                    f"{preview}{'...' if len(pfc_text) > 120 else ''}")
            await self._step()

            if not plan_signal or plan_signal.type != SignalType.PLAN:
                break

            # Only check ACC + confidence for non-fast paths
            if processing_depth == "fast":
                break

            # ── ACC: conflict monitoring (Botvinick 2001)
            conflict = None
            if self._is_active("acc"):
                acc_before = plan_signal
                conflict = await self.acc.process(plan_signal)
                signals_count += 1
                self._apply_gain(self.acc, activation_profile)
                await self._emit("region_activation", "acc", self.acc.activation_level, "active")
                await self._emit("signal_flow", "prefrontal_cortex", "acc", "PLAN", 0.6)
                await self._emit_region_io("acc", acc_before, conflict, "conflict_monitoring")
                await self._step()

            # ── Metacognitive confidence check (Fleming 2012)
            confidence = (plan_signal.metadata.get("metacognition", {})
                         .get("confidence", 0.7))

            # Decide: accept or re-deliberate?
            should_reprocess = (
                reprocess_iter < MAX_REPROCESS
                and (
                    (conflict and conflict.type == SignalType.CONFLICT_DETECTED)
                    or confidence < REPROCESS_CONFIDENCE_THRESHOLD
                )
            )

            if should_reprocess:
                if conflict and conflict.type == SignalType.CONFLICT_DETECTED:
                    conflict_score = conflict.payload.get("conflict_score", 0)
                    self.neuro_ctrl.on_conflict(conflict_score)
                    await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())
                    input_signal.metadata["acc_feedback"] = conflict.payload
                input_signal.metadata["reprocess_reason"] = (
                    "low_confidence" if confidence < REPROCESS_CONFIDENCE_THRESHOLD
                    else "conflict_detected"
                )
                input_signal.metadata["previous_confidence"] = confidence
                await self._emit("region_processing", "acc", "executive",
                    f"Reprocessing iter {reprocess_iter + 1}: "
                    f"confidence={confidence:.2f}, conflict={bool(conflict and conflict.type == SignalType.CONFLICT_DETECTED)}")
                continue  # Re-enter PFC with feedback

            # Accepted — break out
            break

        # PFC → Amygdala top-down regulation (Ochsner & Gross 2005)
        # If PFC produces a calm, rational response, dampen emotional arousal
        if input_signal.emotional_tag and input_signal.emotional_tag.arousal > 0.3:
            # PFC regulation strength depends on network mode
            # ECN mode = strong regulation, CREATIVE = weak, DMN = moderate
            network_mode = self.network_ctrl.current_mode.value
            regulation = {"executive_control": 0.4, "creative": 0.1, "default_mode": 0.25}
            reg_strength = regulation.get(network_mode, 0.2)
            dampened_arousal = input_signal.emotional_tag.arousal * (1.0 - reg_strength)
            dampened_valence = input_signal.emotional_tag.valence * (1.0 - reg_strength * 0.5)
            input_signal.emotional_tag = EmotionalTag(
                valence=dampened_valence, arousal=dampened_arousal
            )

        # Corpus Callosum: inter-hemisphere integration (Gazzaniga 2005)
        if plan_signal and self.pfc._left_activation > 0 and self.pfc._right_activation > 0:
            plan_signal.metadata["left_result"] = {
                "activation": self.pfc._left_activation,
                "mode": "analytical",
                "confidence": self.pfc._left_activation,
            }
            plan_signal.metadata["right_result"] = {
                "activation": self.pfc._right_activation,
                "mode": "holistic",
                "confidence": self.pfc._right_activation,
            }
            cc_before = Signal(type=plan_signal.type, source=plan_signal.source, payload=dict(plan_signal.payload))
            plan_signal = await self.corpus_callosum.process(plan_signal)
            signals_count += 1
            await self._emit_region_io("corpus_callosum", cc_before, plan_signal, "inter_hemisphere_integration")
            await self._emit("region_activation", "corpus_callosum", self.corpus_callosum.activation_level, "active")
            await self._emit("signal_flow", "prefrontal_cortex", "corpus_callosum", "PLAN", 0.6)
            await self._step()

        if plan_signal and plan_signal.payload.get("from_cache"):
            result.from_cache = True

        if plan_signal and plan_signal.type == SignalType.PLAN:
            self._route(plan_signal)

            if processing_depth != "fast":
                # ── Basal Ganglia: Go/NoGo action selection (Mink 1996)
                plan_signal.metadata["neuromodulators"] = self.neuromodulators.snapshot()
                if cached_procedure:
                    plan_signal.metadata["cached_procedure"] = cached_procedure

                action_signal = None
                if self._is_active("basal_ganglia"):
                    bg_before = plan_signal
                    action_signal = await self.basal_ganglia.process(plan_signal)
                    signals_count += 1
                    self._apply_gain(self.basal_ganglia, activation_profile)
                    await self._emit("region_activation", "basal_ganglia", self.basal_ganglia.activation_level, "active")
                    await self._emit("signal_flow", "acc", "basal_ganglia", "PLAN", 0.5)
                    await self._emit_region_io("basal_ganglia", bg_before, action_signal, "action_selection")
                    await self._step()

                if action_signal and action_signal.type == SignalType.ACTION_SELECTED:
                    # ── Cerebellum: forward model prediction (Ito 2008)
                    if self._is_active("cerebellum"):
                        cb_before = action_signal
                        action_signal = await self.cerebellum.process(action_signal)
                        signals_count += 1
                        self._apply_gain(self.cerebellum, activation_profile)
                        await self._emit("region_activation", "cerebellum", self.cerebellum.activation_level, "active")
                        await self._emit("signal_flow", "basal_ganglia", "cerebellum", "ACTION_SELECTED", 0.5)
                        await self._emit_region_io("cerebellum", cb_before, action_signal, "forward_model_prediction")
                        await self._step()

                    # ── Execute (Tool Execution Loop)
                    action = action_signal.payload.get("action", {})
                    tool_name = action.get("tool", "respond")

                    # Tool execution loop: iterate until "respond" or max iterations
                    tool_iteration = 0
                    while (
                        tool_name != "respond"
                        and self.tool_registry
                        and self.tool_registry.has(tool_name)
                        and tool_iteration < self.max_tool_iterations
                    ):
                        tool_iteration += 1
                        tool_params = action.get("args", {})
                        logger.info(
                            "[Pipeline] Tool call %d/%d: %s(%s)",
                            tool_iteration, self.max_tool_iterations, tool_name, list(tool_params.keys()),
                        )
                        await self._emit("broadcast", f"tool_executing:{tool_name}", "pipeline")

                        # Execute tool via registry
                        tool_result_str = await self.tool_registry.execute(tool_name, tool_params)
                        result.actions_taken.append({
                            **action,
                            "result": tool_result_str[:500],
                            "iteration": tool_iteration,
                        })
                        await self._emit("broadcast", f"tool_result:{tool_name}", "pipeline")

                        # Feed tool result back to PFC for next decision
                        tool_feedback = Signal(
                            type=SignalType.TOOL_RESULT,
                            source="tool_executor",
                            payload={
                                "tool": tool_name,
                                "params": tool_params,
                                "result": tool_result_str,
                                "iteration": tool_iteration,
                                "original_input": text,
                                "previous_plan": plan_signal.payload if plan_signal else {},
                            },
                        )
                        # PFC re-plans with tool result context
                        plan_signal = await self.pfc.process(tool_feedback)
                        signals_count += 1

                        if not plan_signal or plan_signal.type != SignalType.PLAN:
                            break

                        # Basal Ganglia re-selects next action
                        plan_signal.metadata["neuromodulators"] = self.neuromodulators.snapshot()
                        action_signal = await self.basal_ganglia.process(plan_signal)
                        signals_count += 1

                        if not action_signal or action_signal.type != SignalType.ACTION_SELECTED:
                            # No further action — extract response from plan
                            for act in plan_signal.payload.get("actions", []):
                                resp = act.get("args", {}).get("text")
                                if resp:
                                    result.response = resp
                                    break
                            break

                        action = action_signal.payload.get("action", {})
                        tool_name = action.get("tool", "respond")

                    # Final: extract text response
                    if tool_name == "respond" or not (self.tool_registry and self.tool_registry.has(tool_name)):
                        result.response = action.get("args", {}).get("text", result.response or "Action executed")
                        result.actions_taken.append(action)

                    if tool_iteration > 0:
                        logger.info("[Pipeline] Tool loop completed after %d iterations", tool_iteration)

                    await self._emit("broadcast", "action_executed", "pipeline")

                    # ── Cerebellum: evaluate executed action
                    go_score = action_signal.payload.get("go_score", 0.7)
                    em_arousal = input_signal.emotional_tag.arousal if input_signal.emotional_tag else 0.0
                    pred_error = max(0.05, (1.0 - go_score) * 0.5 + em_arousal * 0.15)
                    pred = action_signal.payload.get("predicted_outcome", "success")
                    actual_outcome = "success" if pred_error < 0.3 else "failure"
                    self.neuro_ctrl.on_prediction_error(pred_error, pred, actual_outcome)
                    await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

                    # ── Procedural learning (Graybiel 2008, Fitts 1967)
                    reward = self.neuromodulators.reward_signal
                    text_len = len(text.split())
                    logger.info("Procedural gate (action path): pred_error=%.3f, reward=%.3f, words=%d", pred_error, reward, text_len)
                    if pred_error < MINOR_ERROR_THRESHOLD and reward > 0.2 and text_len >= 4:
                        try:
                            existing = await self.memory.procedural.match(text)
                            if existing:
                                await self.memory.procedural.record_execution(existing["id"], success=True)
                                logger.info("Procedural: reinforced '%s'", existing["trigger_pattern"])
                            else:
                                eval_result = await self.pfc.evaluate_procedural(
                                    input_text=text,
                                    response_text=result.response,
                                    comprehension=comprehension,
                                )
                                if eval_result:
                                    await self.memory.procedural.save(
                                        trigger_pattern=eval_result["trigger"],
                                        strategy=eval_result["strategy"],
                                        action_sequence=[action],
                                    )
                                    logger.info("Procedural: saved '%s'", eval_result["trigger"])
                                else:
                                    logger.info("Procedural: LLM returned None (not a pattern)")
                        except Exception as e:
                            logger.warning("Procedural learning failed: %s", e, exc_info=True)

                    self.neuro_ctrl.on_reward_outcome(success=(pred_error < 0.3))
                    await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

                else:
                    # No action selected — still extract response from plan
                    if plan_signal:
                        for act in plan_signal.payload.get("actions", []):
                            resp = act.get("args", {}).get("text")
                            if resp:
                                result.response = resp
                                break

                    # ── Cerebellum: evaluate result (model learning)
                    predicted = action_signal.payload.get("predicted_outcome", "success")
                    tool_name = action.get("tool", "unknown")
                    await self._emit("signal_flow", "basal_ganglia", "cerebellum", "ACTION_RESULT", 0.5)

                    # Compute prediction error dynamically based on action confidence
                    # and emotional context (McGaugh 2004: arousal modulates error signal)
                    action_confidence = action_signal.payload.get("go_score", 0.7)
                    emotional_arousal = (
                        input_signal.emotional_tag.arousal if input_signal.emotional_tag else 0.0
                    )
                    # Higher confidence + calm = lower error; lower confidence + aroused = higher error
                    # Range: ~0.05 (high confidence, calm) to ~0.45 (low confidence, aroused)
                    dynamic_error = max(0.05, (1.0 - action_confidence) * 0.5 + emotional_arousal * 0.15)

                    result_signal = Signal(
                        type=SignalType.ACTION_RESULT,
                        source="executor",
                        payload={
                            "predicted": predicted,
                            "actual": "success",
                            "error": round(dynamic_error, 3),
                            "tool": tool_name,
                        },
                    )
                    error_signal = None
                    if self._is_active("cerebellum"):
                        error_signal = await self.cerebellum.process(result_signal)
                        signals_count += 1

                    # Neuromodulator: prediction error → DA (Schultz 1997)
                    pred = action_signal.payload.get("predicted_outcome", "success")
                    pred_error = float(result_signal.payload.get("error", 0.05))
                    actual_outcome = "success" if pred_error < 0.3 else "failure"
                    self.neuro_ctrl.on_prediction_error(pred_error, pred, actual_outcome)
                    await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

                    # ── VTA: dopamine processing (Schultz 1997)
                    if error_signal and error_signal.type == SignalType.PREDICTION_ERROR:
                        await self.vta.process(error_signal)
                        signals_count += 1
                        self._apply_gain(self.vta, activation_profile)
                        await self._emit("region_activation", "vta", self.vta.activation_level, "active")
                        await self._step()

                    # ── Procedural learning (Graybiel 2008, Fitts 1967)
                    reward = self.neuromodulators.reward_signal
                    text_len = len(text.split())
                    logger.info("Procedural gate (else path): pred_error=%.3f (<%.1f?), reward=%.3f (>0.2?), words=%d (>=4?)",
                                pred_error, MINOR_ERROR_THRESHOLD, reward, text_len)
                    if pred_error < MINOR_ERROR_THRESHOLD and reward > 0.2 and text_len >= 4:
                        try:
                            existing = await self.memory.procedural.match(text)
                            if existing:
                                await self.memory.procedural.record_execution(
                                    existing["id"], success=True,
                                )
                                logger.info("Procedural: reinforced existing '%s'", existing["trigger_pattern"])
                            else:
                                eval_result = await self.pfc.evaluate_procedural(
                                    input_text=text,
                                    response_text=result.response,
                                    comprehension=comprehension,
                                )
                                if eval_result:
                                    await self.memory.procedural.save(
                                        trigger_pattern=eval_result["trigger"],
                                        strategy=eval_result["strategy"],
                                        action_sequence=[action],
                                    )
                                    logger.info("Procedural: saved new '%s' -> '%s'",
                                                eval_result["trigger"], eval_result["strategy"])
                                else:
                                    logger.info("Procedural: LLM returned None (not a pattern)")
                        except Exception as e:
                            logger.warning("Procedural learning failed: %s", e, exc_info=True)

                    # Neuromodulator: reward outcome → 5-HT (Doya 2002)
                    self.neuro_ctrl.on_reward_outcome(success=(pred_error < 0.3))
                    await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())

                    # ── ACC: evaluate outcome (error accumulation)
                    acc_result = None
                    if self._is_active("acc"):
                        acc_result = await self.acc.process(result_signal)
                        signals_count += 1
                        await self._emit("signal_flow", "cerebellum", "acc", "ACTION_RESULT", 0.4)

                        if error_signal and error_signal.type == SignalType.PREDICTION_ERROR:
                            await self.acc.process(error_signal)
                            signals_count += 1

                    if acc_result and acc_result.type == SignalType.STRATEGY_SWITCH:
                        await self.pfc.process(acc_result)
                        signals_count += 1

            else:
                # Fast path: skip ACC/BG/Cerebellum (automatic processing)
                # PFC already used procedural cache → direct to speech production
                if plan_signal:
                    for act in plan_signal.payload.get("actions", []):
                        resp = act.get("args", {}).get("text")
                        if resp:
                            result.response = resp
                            break
                await self._emit("region_processing", "prefrontal_cortex", "executive",
                    "Fast path: automatic processing (Schneider & Shiffrin 1977)")

        # ── Immediate broadcast: PFC response available NOW ──
        # Send to dashboard via WebSocket so user sees the response instantly,
        # before Broca, encoding, or consolidation run.
        if result.response:
            await self._emit("broadcast", result.response, "pipeline")

        # ══════════════════════════════════════════════════════════════
        # Phase 4: Memory Encoding (McGaugh 2004, Hasselmo 2006)
        # RUNS IN BACKGROUND — encoding does not block response delivery.
        # The response is available; hippocampal encoding proceeds async.
        # ══════════════════════════════════════════════════════════════

        etag = input_signal.emotional_tag
        emotional_tag_dict = None
        if etag and (etag.valence != 0 or etag.arousal != 0):
            emotional_tag_dict = {
                "valence": etag.valence,
                "arousal": etag.arousal,
            }

        encode_modality = "visual" if image else "auditory" if audio else "verbal"

        # Build structured content from pipeline context
        extracted = plan_signal.metadata.get("extracted_entities", {}) if plan_signal else {}
        response_text = result.response or ""

        is_fallback = response_text.startswith("Processing: ")
        content_parts = [f"User ({comprehension.get('intent', 'statement')}): {text}"]
        if not is_fallback and response_text and response_text != "Action executed":
            content_parts.append(f"AI: {response_text}")
        encode_content = "\n".join(content_parts)

        encode_entities = {
            "intent": comprehension.get("intent", "statement"),
            "keywords": comprehension.get("keywords", []),
            "complexity": comprehension.get("complexity", "simple"),
            "extracted_entities": extracted.get("entities", []),
            "extracted_relations": extracted.get("relations", []),
            "input": text,
            "output": "" if is_fallback else response_text,
            "network_mode": self.network_ctrl.current_mode.value,
        }

        # New metadata from pipeline phases
        if input_signal.metadata.get("interoceptive_state"):
            encode_entities["interoceptive_state"] = input_signal.metadata["interoceptive_state"]
        if input_signal.metadata.get("attention_weight"):
            encode_entities["attention_weight"] = input_signal.metadata["attention_weight"]
        if plan_signal and plan_signal.metadata.get("metacognition"):
            encode_entities["metacognition"] = plan_signal.metadata["metacognition"]
        prediction_surprise = input_signal.metadata.get("prediction_surprise")
        if prediction_surprise is not None:
            encode_entities["prediction_surprise"] = prediction_surprise

        # ACh (learning_rate) modulates encoding strength (Hasselmo 2006)
        lr = self.neuromodulators.learning_rate

        # Prediction surprise boosts encoding (novel = encode stronger, Friston 2005)
        surprise = input_signal.metadata.get("prediction_surprise", 0.5)
        if surprise > 0.6:
            lr = min(1.0, lr * (1.0 + (surprise - 0.5) * 0.5))

        # ── Background post-response work ──────────────────────────────
        # Capture all variables needed, then schedule as background task.
        # Response is returned immediately; encoding + KG + consolidation
        # proceed without blocking the user.

        _plan_signal = plan_signal
        _comprehension = dict(comprehension)
        _input_text = text
        _hypo_result_ref = [None]  # mutable ref for hypothalamus result

        async def _background_post_response() -> None:
            """Phase 4 encoding + KG storage + Phase 5 consolidation (non-blocking)."""
            try:
                # ── Phase 4: Hippocampal encoding ──
                await self.memory.encode(
                    content=encode_content,
                    entities=encode_entities,
                    emotional_tag=emotional_tag_dict,
                    learning_rate=lr,
                    modality=encode_modality,
                )
                await self._emit("signal_flow", "prefrontal_cortex", "hippocampus", "ENCODE", 0.6)
                await self._emit("region_activation", "hippocampus", 0.7, "active")
                await self._emit("region_processing", "hippocampus", "phase_4",
                    f"Encoded to hippocampus: {encode_modality} modality, strength={lr:.2f}")

                # ── Semantic fact storage (Eichenbaum 2000) ──
                if _plan_signal:
                    _extracted = _plan_signal.metadata.get("extracted_entities", {})
                    kg_entities = _extracted.get("entities", [])
                    about_user = _extracted.get("about_user", [])
                    knowledge = _extracted.get("knowledge", [])
                    logger.info("KG extraction: %d entities, %d about_user, %d knowledge from PFC",
                                len(kg_entities), len(about_user), len(knowledge))

                    kw = _comprehension.get("keywords", [])
                    if kw and not kg_entities:
                        kg_entities = kw
                    if kw and not about_user and not knowledge and _comprehension.get("intent"):
                        intent = _comprehension["intent"]
                        for keyword in kw[:3]:
                            about_user.append(["user", intent, keyword, 0.6, "ACTION"])

                    if about_user:
                        facts_au = [f"{r[0]} {r[1].replace('_', ' ')} {r[2]}" for r in about_user if len(r) >= 3]
                        await self.memory.store_semantic_facts(
                            entities=kg_entities, relations=about_user,
                            facts=facts_au if facts_au else None, origin="agent_about_user",
                        )
                        await self._store_identity_facts_realtime(about_user, source="pfc_extraction")

                    if knowledge:
                        facts_k = [f"{r[0]} {r[1].replace('_', ' ')} {r[2]}" for r in knowledge if len(r) >= 3]
                        await self.memory.store_semantic_facts(
                            entities=kg_entities, relations=knowledge,
                            facts=facts_k if facts_k else None, origin="agent_knowledge",
                        )

                # User-input fact extraction (LLM call — runs in background)
                # Stored as user_input ONLY — this is the ground truth of what
                # the user said. agent_about_user comes from PFC extraction
                # (what the agent understood). Sync rate = gap between them.
                if self.pfc.llm_provider and _comprehension.get("intent"):
                    try:
                        user_extract = await self._extract_user_facts(_input_text, _comprehension)
                        if user_extract:
                            u_ents = user_extract.get("entities", [])
                            u_rels = user_extract.get("relations", [])
                            await self.memory.store_semantic_facts(
                                entities=u_ents, relations=u_rels, origin="user_input",
                            )
                            await self._store_identity_facts_realtime(u_rels, source="user_input")
                    except Exception:
                        logger.warning("User fact extraction failed", exc_info=True)

                # ── Phase 5: Consolidation ──
                staging_count = await self.memory.staging.count_unconsolidated()
                hypo_result = _hypo_result_ref[0]
                should_consolidate = (
                    (hypo_result and hypo_result.type == SignalType.CONSOLIDATION_TRIGGER)
                    or await self.memory.consolidation.should_consolidate()
                )
                logger.info("Consolidation check: %d unconsolidated, should_consolidate=%s",
                             staging_count, should_consolidate)

                try:
                    staging_snapshot = await self.memory.staging.get_unconsolidated()
                except Exception:
                    staging_snapshot = []

                if should_consolidate:
                    await self._emit("region_processing", "hippocampus", "phase_5",
                        "Consolidation triggered — transferring memories")
                    await self.memory.consolidate()

                if staging_snapshot:
                    try:
                        from brain_agent.memory.narrative_consolidation import narrative_consolidate
                        llm = getattr(self.pfc, 'llm_provider', None) or self._llm_provider
                        if not llm:
                            logger.error("Phase 5: NO LLM provider for narrative consolidation.")
                        else:
                            success = await narrative_consolidate(
                                staging_snapshot, llm,
                                semantic_store=self.memory.semantic,
                            )
                            logger.info("Narrative consolidation result: %s", success)
                            if success:
                                self.tpj.reload_schema()
                                self.mpfc.reload_schema()
                    except Exception as e:
                        logger.warning("Narrative consolidation failed: %s", e)

                # Layer 3: Dreaming
                if self.dreaming.should_dream():
                    try:
                        from brain_agent.memory.narrative_consolidation import _read_file, _write_file
                        promotion_text = await self.dreaming.run_cycle()
                        if promotion_text:
                            current_memory = _read_file("memory/MEMORY.md")
                            _write_file("memory/MEMORY.md",
                                        current_memory.rstrip() + "\n\n" + promotion_text + "\n")
                    except Exception as e:
                        logger.warning("Dreaming cycle failed: %s", e)

                # Neuromodulator decay + brain state save
                self.neuro_ctrl.decay()
                await self._emit("neuromodulator_update", **self.neuromodulators.snapshot())
                try:
                    await self.save_brain_state()
                except Exception:
                    pass

            except Exception as e:
                logger.error("Background post-response failed: %s", e, exc_info=True)

        # Mark encoding as scheduled (flag set immediately, encoding runs in background)
        result.memory_encoded = True

        # Schedule background work — does NOT block response return
        import asyncio as _aio
        _aio.create_task(_background_post_response())

        # ══════════════════════════════════════════════════════════════
        # Phase 7: Speech Production (Levelt 1989)
        # Runs in BACKGROUND — Broca refines, M1 formats.
        # If Broca produces a different response, broadcast via WS.
        # ══════════════════════════════════════════════════════════════

        _plan_signal_broca = plan_signal
        _pfc_response = result.response  # snapshot before Broca
        _activation_profile = activation_profile
        _comprehension_broca = comprehension

        async def _background_broca() -> None:
            """Phase 7: Broca + Motor Cortex (non-blocking)."""
            try:
                if not _plan_signal_broca:
                    return

                _plan_signal_broca.metadata["comprehension"] = _comprehension_broca
                broca_before = _plan_signal_broca

                if should_full_process("broca_area", _activation_profile):
                    broca_result = await self.broca.process(_plan_signal_broca)
                else:
                    self.broca._format_heuristic(_plan_signal_broca)
                    self.broca.emit_activation(0.4)
                    broca_result = _plan_signal_broca

                # Extract Broca's refined response
                refined_response = None
                if broca_result and self.broca.llm_provider and should_full_process("broca_area", _activation_profile):
                    for act in broca_result.payload.get("actions", []):
                        new_text = act.get("args", {}).get("text")
                        if new_text:
                            refined_response = new_text
                            break
                    if not refined_response:
                        new_resp = broca_result.payload.get("response_text")
                        if new_resp:
                            refined_response = new_resp

                self._apply_gain(self.broca, _activation_profile)
                await self._emit_region_io("broca", broca_before, broca_result, "language_formulation")
                await self._emit("region_activation", "broca", self.broca.activation_level, "active")
                await self._emit("region_processing", "broca", "phase_7", "Language formulation complete")

                # M1: final output formatting
                await self.motor_cortex.process(_plan_signal_broca)
                await self._emit("region_activation", "motor_cortex", self.motor_cortex.activation_level, "active")

                # If Broca refined the response and it differs, broadcast update
                if refined_response and refined_response != _pfc_response:
                    await self._emit("broadcast", refined_response, "broca_refined")

                # Predictive coding: store prediction for next input
                final_resp = refined_response or _pfc_response
                if final_resp:
                    try:
                        pred_emb = self.memory._embed_fn(final_resp)
                        self.predictor.store_prediction(pred_emb)
                    except Exception:
                        pass

            except Exception as e:
                logger.error("Background Broca failed: %s", e, exc_info=True)

        _aio.create_task(_background_broca())

        # ── GWT Broadcast (Baars 1988) — lightweight, in-memory
        self.workspace.submit(
            Signal(
                type=SignalType.GWT_BROADCAST,
                source="pipeline",
                payload={"status": "task_complete", "result": result.response},
            ),
            salience=0.7,
            goal_relevance=0.8,
        )
        broadcast = self.workspace.compete()
        if broadcast:
            await self.salience.process(broadcast)
            signals_count += 1
        await self._emit("network_switch", "ECN", "DMN", "task_complete")

        # ── Hypothalamus: quick resource check (passes result to background)
        error_rate = (
            self.acc.error_accumulator / self.acc.strategy_switch_threshold
            if self.acc.strategy_switch_threshold > 0
            else 0.0
        )
        hypo_signal = Signal(
            type=SignalType.RESOURCE_STATUS,
            source="pipeline",
            payload={"pending_requests": 0, "staging_count": 0, "error_rate": min(1.0, error_rate)},
        )
        hypo_result = await self.hypothalamus.process(hypo_signal)
        signals_count += 1
        _hypo_result_ref[0] = hypo_result  # Pass to background task

        self.neuro_ctrl.on_system_state(pending_requests=0, error_rate=min(1.0, error_rate))

        # ── Return response IMMEDIATELY ──
        # Phase 4 (encoding), KG storage, Phase 5 (consolidation),
        # Phase 7 (Broca refinement), dreaming, neuromodulator decay,
        # and brain state persistence all proceed in background.
        result.network_mode = self.network_ctrl.current_mode.value
        result.signals_processed = signals_count
        return result
