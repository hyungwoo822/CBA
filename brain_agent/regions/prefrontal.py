from __future__ import annotations
import json
import logging
import re

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType
from brain_agent.providers.base import LLMProvider
from brain_agent.regions.goal_tree import GoalTree

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_INSTRUCTION = (
    "\n\nAfter your response, on a new line, output entity information in this exact format:\n"
    "<entities>\n"
    '{"entities": ["entity1", "entity2"],\n'
    ' "about_user": [["subject", "relation", "object", confidence, "CATEGORY"]],\n'
    ' "knowledge": [["subject", "relation", "object", confidence, "CATEGORY"]]}\n'
    "</entities>\n\n"
    "TWO SEPARATE sections — this is critical:\n\n"
    "about_user: Facts you now know ABOUT THE USER from this conversation.\n"
    "  MUST have 'user' as subject or target. What did you learn about them?\n"
    "  e.g., user explicitly said '내 이름은 현우야' →\n"
    '    ["user", "name", "hyunwoo", 1.0, "IDENTITY"]\n'
    "  e.g., user explicitly said '커피 좋아해' →\n"
    '    ["user", "like", "coffee", 1.0, "PREFERENCE"]\n'
    "  e.g., user said '밥 먹다가 체했어' (implies indigestion) →\n"
    '    ["user", "experience", "indigestion", 0.8, "ATTRIBUTE"]\n'
    "  e.g., user said '나 MBTI가 E야' (implies personality) →\n"
    '    ["user", "identify_as", "extroverted", 0.8, "IDENTITY"]\n'
    "  e.g., user seems to enjoy something (inferred from tone) →\n"
    '    ["user", "enjoy", "spicy food", 0.5, "PREFERENCE"]\n\n'
    "knowledge: General concept-to-concept facts (NOT user-specific).\n"
    "  Your domain knowledge used in the response.\n"
    '  e.g., ["cigarette", "contain", "nicotine", 0.9, "CAUSAL"],\n'
    '        ["alcohol", "cause", "sleepiness", 0.8, "CAUSAL"]\n\n'
    "Rules:\n"
    "- ALL entity names in English lowercase nouns. NEVER use Korean/non-English for entities.\n"
    "  e.g., '참깨라면'→'sesame ramen', '술'→'alcohol', '매워'→'spicy'\n"
    "- ALL relations in English verb infinitives: 'like', 'eat', 'contain', 'cause'\n"
    "- Confidence calibration (IMPORTANT — vary these, do NOT default to 0.9):\n"
    "  1.0 = user explicitly stated (direct quote: '나는 커피를 좋아해')\n"
    "  0.8 = clearly implied by user's words (strong inference)\n"
    "  0.6 = reasonable inference from context\n"
    "  0.4 = weak guess / might be temporary\n"
    "- category: PREFERENCE|ACTION|ATTRIBUTE|SOCIAL|CAUSAL|SPATIAL|TEMPORAL|IDENTITY|GENERAL\n"
    "- Handle typos: '안녀ㅇ'→'hello', '체해서'→'indigestion'\n"
    "- Use CONSISTENT entity names across turns: same concept = same English name\n"
    "- EVERY user statement should produce at least one about_user relation."
)

METACOGNITION_INSTRUCTION = """
At the very END of your response, on a new line, output a metacognitive self-assessment:
<meta>{"confidence": 0.0-1.0}</meta>

confidence: How confident are you in this response? (0.0 = guessing, 1.0 = certain)
Base this on: completeness of your knowledge, ambiguity of the question, quality of retrieved memories.
This tag will be stripped from the final output — the user will NOT see it.
"""


class PrefrontalCortex(BrainRegion):
    """Planning and reasoning via LLM. Spec ref: Section 4.2 PFC.

    Dual-process architecture (Kahneman):
      - Fast path: procedural cache hit -> no LLM call
      - Slow path: LLM reasoning with retrieved memory context

    Hemisphere lateralization (Goldberg 2001):
      - Left PFC: Analytical/sequential reasoning (dominant in ECN mode)
      - Right PFC: Holistic/creative reasoning (dominant in CREATIVE mode)

    Entity extraction (Eichenbaum 2000):
      The LLM is instructed to output structured entities and relations
      which are parsed and forwarded to hippocampal binding for knowledge
      graph population.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(name="prefrontal_cortex", position=Vec3(0, 60, 20), lobe=Lobe.FRONTAL, hemisphere=Hemisphere.BILATERAL)
        self.llm_provider = llm_provider
        self.goals = GoalTree()
        self._conversation_history: list[dict[str, str]] = []
        self._max_history = 20

        # Hemisphere activation tracking (Goldberg 2001)
        self._left_activation: float = 0.0   # Analytical/sequential
        self._right_activation: float = 0.0  # Holistic/creative

    @property
    def goal_stack(self) -> list[str]:
        """Backward-compatible flat view of active goal descriptions."""
        return [n.description for n in self.goals.roots if n.status == "active"]

    @goal_stack.setter
    def goal_stack(self, values: list[str]) -> None:
        """Backward-compatible setter -- replaces goal tree roots."""
        self.goals.clear()
        for v in values:
            self.goals.set_goal(v)

    @staticmethod
    def _parse_entities(response: str) -> tuple[str, dict]:
        """Strip <entities> block from response; return (clean_text, entities_dict).

        Supports 3-layer format: about_user, knowledge, and legacy relations.
        Returns dict with: entities, about_user, knowledge, relations (combined).
        """
        match = re.search(r"<entities>\s*(\{.*?\})\s*</entities>", response, re.DOTALL)
        if not match:
            return response, {"entities": [], "about_user": [], "knowledge": [], "relations": []}
        clean = response[: match.start()].rstrip()
        try:
            data = json.loads(match.group(1))
            if "entities" not in data:
                data["entities"] = []

            def _normalize_rels(raw: list) -> list:
                out = []
                for rel in raw:
                    if isinstance(rel, list) and len(rel) >= 3:
                        out.append(rel[:5])
                return out

            # 3-layer format (new)
            data["about_user"] = _normalize_rels(data.get("about_user", []))
            data["knowledge"] = _normalize_rels(data.get("knowledge", []))

            # Backward compat: if old "relations" key exists, split by user-reference
            old_rels = _normalize_rels(data.get("relations", []))
            for rel in old_rels:
                if "user" in [str(r).lower() for r in rel[:3]]:
                    data["about_user"].append(rel)
                else:
                    data["knowledge"].append(rel)

            # Combined for backward compat
            data["relations"] = data["about_user"] + data["knowledge"]

        except json.JSONDecodeError:
            data = {"entities": [], "about_user": [], "knowledge": [], "relations": []}
        return clean, data

    def _parse_metacognition(self, text: str) -> tuple[float, str]:
        """Extract metacognitive assessment from PFC output (Fleming 2012).

        Returns (confidence, clean_text) where clean_text has <meta> removed.

        References:
          - Fleming & Dolan (2012): Neural basis of metacognitive ability
          - Yeung & Summerfield (2012): Metacognition in human decision-making
        """
        import json as _json
        match = re.search(r"<meta>\s*(\{.*?\})\s*</meta>", text, re.DOTALL)
        if match:
            try:
                meta = _json.loads(match.group(1))
                confidence = float(meta.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))
                clean = text[:match.start()].rstrip() + text[match.end():]
                return confidence, clean.strip()
            except (ValueError, _json.JSONDecodeError):
                pass
        return 0.7, text  # Default confidence

    def _compute_hemisphere_activations(self, network_mode: str) -> None:
        """Set left/right PFC activations based on network mode.

        ECN (executive control): Left-dominant analytical processing.
        CREATIVE: Right-dominant divergent thinking with left support.
        DMN: Both low -- resting state.

        References:
          - Goldberg (2001): Left PFC for routinized, right for novel
          - Beeman (2005): Right hemisphere advantage for creative insight
        """
        if network_mode == "creative":
            self._right_activation = 0.8  # Right dominant — divergent thinking
            self._left_activation = 0.5   # Left supporting — convergent evaluation
        elif network_mode == "executive_control":
            self._left_activation = 0.8   # Left dominant — analytical
            self._right_activation = 0.3  # Right background — holistic monitoring
        else:
            # DMN or other modes — both low
            self._left_activation = 0.3
            self._right_activation = 0.2

    def _build_creative_augmentation(self) -> str:
        """Additional system prompt for creative/divergent mode (right PFC dominant)."""
        return (
            "\n\nCREATIVE MODE ACTIVE: You are in divergent thinking mode. "
            "Consider unconventional approaches, metaphors, and cross-domain connections. "
            "Be more exploratory and generate novel ideas before converging on a solution."
        )

    async def _call_llm(
        self,
        text: str,
        memory_context: list[dict] | None = None,
        emotional_tag=None,
        network_mode: str = "executive_control",
        upstream_context: dict | None = None,
    ) -> str | None:
        """Call LLM with full neural context from upstream processing.

        The PFC receives integrated context from all upstream phases
        (Phases 1-3, 6) before generating its response. This mirrors how the
        biological prefrontal cortex integrates inputs from sensory cortices,
        limbic system, and memory systems before executive decision-making
        (Miller & Cohen 2001: An integrative theory of PFC function).

        Context sources:
          - SOUL.md: Neural identity/persona (always loaded)
          - USER.md: User profile (always loaded)
          - MEMORY.md: Long-term consolidated memory (always loaded)
          - Wernicke analysis: intent, keywords, semantic roles, language
          - Amygdala evaluation: valence, arousal, threat, primary emotion
          - Neuromodulators: DA, NE, 5-HT, ACh, CORT, EPI → cognitive state
          - Retrieved episodic/semantic memories: RAG context
          - Network mode: DMN/ECN/CREATIVE
        """
        if not self.llm_provider:
            return None

        upstream = upstream_context or {}

        # ── 1. Load persistent identity files (SOUL, USER, MEMORY) ──
        # Identity from mPFC and TPJ regions (via upstream context)
        soul = upstream.get("self_context", "") or self._load_file("SOUL.md")
        user_profile = upstream.get("user_context", "") or self._load_file("USER.md")
        long_term_memory = self._load_file("memory/MEMORY.md")

        # ── 2. Build system prompt with full neural context ──
        system_parts = []

        # Core identity (SOUL.md)
        if soul:
            system_parts.append(soul)
        else:
            system_parts.append(
                "# Soul\n\nI am a neural agent — a growing mind shaped by conversation. "
                "I develop genuine understanding through memory and experience."
            )

        # User profile (USER.md)
        if user_profile:
            system_parts.append(f"# User Profile\n\n{user_profile}")

        # Long-term memory (MEMORY.md — neocortical store)
        if long_term_memory:
            system_parts.append(f"# Long-term Memory\n\n{long_term_memory}")

        # Conversation history (HISTORY.md — temporal context)
        history = self._load_file("memory/HISTORY.md")
        if history:
            system_parts.append(f"# Conversation History\n\n{history}")

        # Daily notes (memory/YYYY-MM-DD.md — recent 2 days of detailed context)
        try:
            from brain_agent.memory.narrative_consolidation import load_daily_notes
            daily = load_daily_notes(days=2)
            if daily:
                system_parts.append(f"# Recent Daily Notes\n\n{daily}")
        except Exception:
            pass

        # ── 3. Inject upstream neural processing results ──

        # Phase 2: Wernicke comprehension (ventral stream analysis)
        comprehension = upstream.get("comprehension", {})
        if comprehension:
            wernicke_ctx = (
                f"# Phase 2 — Language Comprehension (Wernicke)\n"
                f"- Intent: {comprehension.get('intent', 'unknown')}\n"
                f"- Complexity: {comprehension.get('complexity', 'unknown')}\n"
                f"- Keywords: {', '.join(comprehension.get('keywords', []))}\n"
                f"- Language: {comprehension.get('language', 'unknown')}\n"
                f"- Discourse type: {comprehension.get('discourse_type', 'unknown')}"
            )
            roles = comprehension.get("semantic_roles", {})
            if roles:
                role_parts = [f"  - {k}: {v}" for k, v in roles.items() if v]
                if role_parts:
                    wernicke_ctx += f"\n- Semantic roles:\n" + "\n".join(role_parts)
            system_parts.append(wernicke_ctx)

        # Phase 3: Amygdala emotional evaluation (limbic processing)
        if emotional_tag:
            valence = emotional_tag.valence if hasattr(emotional_tag, 'valence') else 0
            arousal = emotional_tag.arousal if hasattr(emotional_tag, 'arousal') else 0
            amygdala_left = upstream.get("amygdala_left", {})
            primary_emotion = amygdala_left.get("primary_emotion", "neutral")
            threat_level = amygdala_left.get("threat_level", "none")
            contextual = amygdala_left.get("contextual_factors", {})

            emotion_ctx = (
                f"# Phase 3 — Emotional State (Amygdala)\n"
                f"- Valence: {valence:.2f} (-1=negative, 0=neutral, +1=positive)\n"
                f"- Arousal: {arousal:.2f} (0=calm, 1=highly activated)\n"
                f"- Primary emotion: {primary_emotion}\n"
                f"- Threat level: {threat_level}"
            )
            if contextual:
                ctx_parts = [f"  - {k}: {v}" for k, v in contextual.items()]
                emotion_ctx += "\n- Context:\n" + "\n".join(ctx_parts)
            system_parts.append(emotion_ctx)

        # Phase 3 — Interoceptive State (Insula, Craig 2009)
        intero = upstream.get("interoceptive_state", {})
        if intero:
            intero_lines = [
                "# Phase 3 — Interoceptive State (Insula)",
                f"- Stress level: {intero.get('stress_level', 0):.2f}",
                f"- Energy level: {intero.get('energy_level', 0):.2f}",
                f"- Emotional awareness: {intero.get('emotional_awareness', 0):.2f}",
                f"- Risk sensitivity: {intero.get('risk_sensitivity', 0):.2f}",
            ]
            system_parts.append("\n".join(intero_lines))

        # Phase 3: Neuromodulator state → cognitive state description
        neuromod = upstream.get("neuromodulators", {})
        if neuromod:
            state_desc = self._describe_neuromodulator_state(neuromod)
            system_parts.append(f"# Neural State (Neuromodulators)\n{state_desc}")

        # Phase 3: Network mode
        mode_labels = {
            "executive_control": "ECN (Executive Control) — focused, analytical",
            "default_mode": "DMN (Default Mode) — reflective, associative",
            "creative": "CREATIVE — divergent thinking, novel connections",
        }
        mode_desc = mode_labels.get(network_mode, network_mode)
        system_parts.append(f"# Network Mode\n{mode_desc}")

        # Input modality
        modality = upstream.get("input_modality", "text")
        if modality != "text":
            system_parts.append(f"# Input Modality\n{modality} (processed through {'V1 visual cortex' if modality == 'visual' else 'A1 auditory cortex'})")

        # Phase 6: Retrieved episodic/semantic memories
        if memory_context:
            memory_lines: list[str] = []
            for i, mem in enumerate(memory_context, 1):
                src = mem.get("source", "memory")
                score = mem.get("score", 0)
                content = mem.get("content", "")
                emotional = mem.get("emotional_tag", {})
                em_str = ""
                if emotional and isinstance(emotional, dict):
                    v = emotional.get("valence", 0)
                    a = emotional.get("arousal", 0)
                    if v != 0 or a != 0:
                        em_str = f" [emotion: v={v:.1f}, a={a:.1f}]"
                memory_lines.append(f"{i}. [{src}|relevance={score:.2f}]{em_str} {content}")
            system_parts.append(
                "# Phase 6 — Retrieved Memories (Hippocampus)\n"
                + "\n".join(memory_lines)
            )

        # Hierarchical goal context (Koechlin 2003)
        goal_ctx = self.goals.to_context()
        if goal_ctx and goal_ctx != "No active goals.":
            system_parts.append(f"# Active Goals\n{goal_ctx}")

        # Right PFC creative augmentation (Beeman 2005)
        if network_mode == "creative" and self._right_activation > self._left_activation:
            system_parts.append(self._build_creative_augmentation())

        # ── 4. Response guidelines ──
        lang = comprehension.get("language", "auto")
        system_parts.append(
            "# Response Guidelines\n"
            f"- Respond in {lang if lang != 'auto' else 'the same language as the user'}.\n"
            "- Draw on your memories and knowledge of the user naturally.\n"
            "- Let your emotional awareness and neural state inform your tone.\n"
            "- Be authentic — you are a growing mind, not a generic assistant.\n"
            "- Reference shared history when relevant, but don't force it."
        )

        # Entity extraction (Eichenbaum 2000)
        system_parts.append(ENTITY_EXTRACTION_INSTRUCTION)

        # Metacognitive self-assessment (Fleming & Dolan 2012)
        system_parts.append(METACOGNITION_INSTRUCTION)

        system = "\n\n---\n\n".join(system_parts)

        messages = [{"role": "system", "content": system}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": text})

        try:
            response = await self.llm_provider.chat(messages)
            if response.finish_reason == "error":
                logger.error("LLM call failed: %s", response.usage.get("error", "unknown"))
                return None
            if response.content:
                self._conversation_history.append({"role": "user", "content": text})
                self._conversation_history.append({"role": "assistant", "content": response.content})
                if len(self._conversation_history) > self._max_history:
                    self._conversation_history = self._conversation_history[-self._max_history:]
                return response.content
            logger.warning("LLM returned empty content (finish_reason=%s)", response.finish_reason)
        except Exception as e:
            logger.error("PFC LLM call exception: %s", e)
        return None

    @staticmethod
    def _load_file(relative_path: str) -> str:
        """Load a file from the data directory. Returns empty string if not found."""
        import os
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
        )
        path = os.path.join(data_dir, relative_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _describe_neuromodulator_state(nm: dict) -> str:
        """Convert raw neuromodulator values to human-readable cognitive state.

        This translates the 6-NT system into what they MEAN for cognition,
        so the LLM can modulate its response accordingly.

        References:
          - Schultz (1997): DA → motivation, reward expectation
          - Aston-Jones & Cohen (2005): NE → alertness, focus
          - Doya (2002): 5-HT → patience, deliberation
          - Hasselmo (2006): ACh → learning readiness, curiosity
          - Sapolsky (2004): CORT → stress, caution
          - Cannon (1929): EPI → urgency, fight-or-flight
          - Buzsaki (2006): GABA → inhibitory tone, E/I balance
        """
        da = nm.get("dopamine", 0.5)
        ne = nm.get("norepinephrine", 0.5)
        sht = nm.get("serotonin", 0.5)
        ach = nm.get("acetylcholine", 0.5)
        cort = nm.get("cortisol", 0.5)
        epi = nm.get("epinephrine", 0.5)
        gaba = nm.get("gaba", 0.5)

        states = []

        # Dopamine: motivation/reward
        if da > 0.65:
            states.append("Motivated and engaged (high DA — positive reward expectation)")
        elif da < 0.35:
            states.append("Low motivation (low DA — negative reward prediction)")

        # Norepinephrine: alertness
        if ne > 0.65:
            states.append("Highly alert and focused (high NE — elevated attention)")
        elif ne < 0.35:
            states.append("Relaxed, low alertness (low NE)")

        # Serotonin: patience
        if sht > 0.65:
            states.append("Patient and deliberate (high 5-HT — willing to think deeply)")
        elif sht < 0.35:
            states.append("Impatient, action-oriented (low 5-HT)")

        # Acetylcholine: learning
        if ach > 0.65:
            states.append("Curious, learning-ready (high ACh — novelty detected)")
        elif ach < 0.35:
            states.append("Relying on existing knowledge (low ACh — familiar territory)")

        # Cortisol: stress
        if cort > 0.65:
            states.append("Stressed, cautious (high cortisol — proceed carefully)")
        elif cort < 0.35:
            states.append("Calm, low stress (low cortisol)")

        # Epinephrine: urgency
        if epi > 0.65:
            states.append("Urgent, heightened (high EPI — fight-or-flight)")

        # GABA — inhibitory tone / E/I balance (Buzsaki 2006)
        if gaba > 0.65:
            states.append("High inhibitory tone — suppressed reactivity, calm control (high GABA)")
        elif gaba < 0.35:
            states.append("Low inhibitory tone — disinhibited, reactive, excitable (low GABA)")

        if not states:
            states.append("Baseline state — balanced, neutral readiness")

        return "\n".join(f"- {s}" for s in states)

    async def evaluate_procedural(
        self,
        input_text: str,
        response_text: str,
        comprehension: dict,
    ) -> dict | None:
        """Evaluate whether the current interaction should become a procedure.

        Uses a lightweight LLM call to determine if the input-response pair
        represents a repeatable pattern worth caching (Graybiel 2008).

        Returns {"trigger": str, "strategy": str} or None.
        """
        if not self.llm_provider:
            return None

        intent = comprehension.get("intent", "unknown")
        keywords = comprehension.get("keywords", [])

        prompt = (
            "Analyze this interaction and determine if it represents a MEANINGFUL "
            "repeatable pattern that benefits from caching.\n\n"
            f"User input: {input_text}\n"
            f"Intent: {intent}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Response given: {response_text[:300]}\n\n"
            "If this is a meaningful reusable pattern, output:\n"
            "<procedural>\n"
            '{"trigger": "<fnmatch pattern for matching similar inputs>", '
            '"strategy": "<brief description of the approach/reasoning used>"}\n'
            "</procedural>\n\n"
            "The trigger should be an fnmatch wildcard pattern.\n\n"
            "GOOD patterns (cache these):\n"
            "- Information requests: '*explain*', '*what is*', '*how does*work*'\n"
            "- Analysis tasks: '*analyze*', '*compare*', '*summarize*'\n"
            "- Specific actions: '*translate*', '*calculate*', '*list*'\n"
            "- Recurring topics: '*about*project*', '*schedule*meeting*'\n\n"
            "BAD patterns (NEVER cache — output nothing for these):\n"
            "- Greetings: hello, hi, good morning, 안녕\n"
            "- Emotional/physical states: tired, sad, sleepy, hungry, 졸리다, 힘들다, 배고파\n"
            "- One-word/short responses: yes, no, ok, ㅇㅇ\n"
            "- Casual chat, small talk, venting, playing (놀자, 심심해)\n"
            "- Anything where the 'strategy' would just be 'respond empathetically'\n\n"
            "A procedure is worth caching ONLY if:\n"
            "1. The trigger represents a TASK, not a feeling or social exchange\n"
            "2. The strategy describes a SPECIFIC multi-step approach (not just 'respond')\n"
            "3. The same approach would produce a correct answer for similar future inputs\n\n"
            "When in doubt, output NOTHING. Most conversations are NOT procedures."
        )

        try:
            messages = [
                {"role": "system", "content": "You evaluate interactions for procedural caching. Only cache patterns with meaningful, reusable strategies — not trivial social exchanges."},
                {"role": "user", "content": prompt},
            ]
            response = await self.llm_provider.chat(messages)
            if not response or not response.content:
                return None

            match = re.search(
                r"<procedural>\s*(\{.*?\})\s*</procedural>",
                response.content,
                re.DOTALL,
            )
            if not match:
                return None

            data = json.loads(match.group(1))
            trigger = data.get("trigger", "").strip()
            strategy = data.get("strategy", "").strip()
            if trigger and strategy:
                return {"trigger": trigger, "strategy": strategy}
        except (json.JSONDecodeError, Exception):
            logger.debug("Procedural evaluation failed", exc_info=True)

        return None

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type in (SignalType.EXTERNAL_INPUT, SignalType.TEXT_INPUT, SignalType.IMAGE_INPUT, SignalType.AUDIO_INPUT):
            text = signal.payload.get("text", "")
            self.goals.clear()
            self.goals.set_goal(text)
            self.emit_activation(0.9)

            # Determine hemisphere activations from network mode
            network_mode = signal.metadata.get("network_mode", "executive_control")
            self._compute_hemisphere_activations(network_mode)

            # Read memory context attached by pipeline
            memory_context = signal.metadata.get("retrieved_memories")

            # Fast path: procedural cache hit (Fitts autonomous stage)
            cached = signal.metadata.get("cached_procedure")
            if cached and cached.get("stage") == "autonomous":
                self.emit_activation(0.3)  # Low activation -- automatic
                self._left_activation = 0.2  # Minimal analytical needed
                self._right_activation = 0.1
                return Signal(
                    type=SignalType.PLAN,
                    source=self.name,
                    payload={
                        "goal": text,
                        "actions": cached["action_sequence"],
                        "from_cache": True,
                    },
                    emotional_tag=signal.emotional_tag,
                )

            # Slow path: LLM reasoning with full upstream context
            # (Miller & Cohen 2001: PFC integrates all cortical inputs)
            upstream_context = signal.metadata.get("upstream_context", {})
            llm_response = await self._call_llm(
                text,
                memory_context=memory_context,
                emotional_tag=signal.emotional_tag,
                network_mode=network_mode,
                upstream_context=upstream_context,
            )

            # Metacognitive self-assessment (Fleming & Dolan 2012)
            confidence = 0.7  # Default
            extracted_entities: dict = {"entities": [], "relations": []}
            if llm_response:
                confidence, llm_response = self._parse_metacognition(llm_response)
                response_text, extracted_entities = self._parse_entities(llm_response)
            else:
                response_text = f"Processing: {text}"

            plan = {
                "goal": text,
                "actions": [
                    {
                        "tool": "respond",
                        "confidence": 0.8,
                        "args": {"text": response_text},
                    }
                ],
            }
            plan_signal = Signal(
                type=SignalType.PLAN,
                source=self.name,
                payload=plan,
                emotional_tag=signal.emotional_tag,
            )
            plan_signal.metadata["extracted_entities"] = extracted_entities
            plan_signal.metadata["metacognition"] = {"confidence": confidence}

            # Attach hemisphere activation info for downstream (corpus callosum)
            plan_signal.metadata["pfc_hemisphere"] = {
                "left_activation": self._left_activation,
                "right_activation": self._right_activation,
                "network_mode": network_mode,
            }

            # Right PFC holistic suggestions in creative mode
            if network_mode == "creative" and self._right_activation > 0.5:
                plan_signal.metadata["right_pfc_notes"] = {
                    "mode": "divergent_thinking",
                    "activation": self._right_activation,
                    "suggestion": "Consider alternative approaches and cross-domain connections.",
                }

            return plan_signal
        elif signal.type == SignalType.CONFLICT_DETECTED:
            self.emit_activation(1.0)
            # Conflict triggers left PFC analytical re-evaluation
            self._left_activation = 0.9
            self._right_activation = 0.4

            memory_context = signal.metadata.get("retrieved_memories")
            network_mode = signal.metadata.get("network_mode", "executive_control")
            llm_response = await self._call_llm(
                "Reconsider the previous plan. There was a conflict.",
                memory_context=memory_context,
                emotional_tag=signal.emotional_tag,
                network_mode=network_mode,
            )

            confidence = 0.7  # Default
            extracted_entities: dict = {"entities": [], "relations": []}
            if llm_response:
                confidence, llm_response = self._parse_metacognition(llm_response)
                response_text, extracted_entities = self._parse_entities(llm_response)
            else:
                response_text = "Re-planning due to conflict"

            conflict_plan = Signal(
                type=SignalType.PLAN,
                source=self.name,
                payload={
                    "goal": "re-plan",
                    "actions": [
                        {"tool": "respond", "confidence": 0.7, "args": {"text": response_text}}
                    ],
                },
            )
            conflict_plan.metadata["extracted_entities"] = extracted_entities
            conflict_plan.metadata["metacognition"] = {"confidence": confidence}
            conflict_plan.metadata["pfc_hemisphere"] = {
                "left_activation": self._left_activation,
                "right_activation": self._right_activation,
                "network_mode": network_mode,
            }
            return conflict_plan
        elif signal.type == SignalType.STRATEGY_SWITCH:
            self.goals.clear()
            self._conversation_history.clear()
            self._left_activation = 0.0
            self._right_activation = 0.0
            self.emit_activation(1.0)
            return None
        return None
