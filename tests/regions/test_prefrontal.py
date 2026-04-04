import pytest
from unittest.mock import AsyncMock
from brain_agent.regions.prefrontal import PrefrontalCortex
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.providers.base import LLMProvider, LLMResponse


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat.return_value = LLMResponse(
        content="Hello! I'm doing well.",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    )
    return provider


@pytest.fixture
def pfc_with_llm(mock_llm_provider):
    return PrefrontalCortex(llm_provider=mock_llm_provider)


@pytest.fixture
def pfc():
    return PrefrontalCortex()


async def test_generates_plan_from_input(pfc):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "read file auth.py"})
    result = await pfc.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    assert result.payload["goal"] == "read file auth.py"
    assert len(result.payload["actions"]) > 0


async def test_plan_action_has_tool_and_confidence(pfc):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "list files"})
    result = await pfc.process(sig)
    action = result.payload["actions"][0]
    assert "tool" in action
    assert "confidence" in action


async def test_preserves_emotional_tag(pfc):
    tag = EmotionalTag(valence=-0.5, arousal=0.8)
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "error occurred"}, emotional_tag=tag)
    result = await pfc.process(sig)
    assert result.emotional_tag is tag


async def test_sets_goal_stack(pfc):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "analyze codebase"})
    await pfc.process(sig)
    assert pfc.goal_stack == ["analyze codebase"]


async def test_emits_high_activation_on_input(pfc):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "task"})
    await pfc.process(sig)
    assert pfc.activation_level == pytest.approx(0.9)


async def test_replans_on_conflict(pfc):
    sig = Signal(type=SignalType.CONFLICT_DETECTED, source="acc",
                 payload={"reason": "conflicting actions"})
    result = await pfc.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    assert pfc.activation_level == pytest.approx(1.0)


async def test_clears_goals_on_strategy_switch(pfc):
    # First set some goals
    input_sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                       payload={"text": "task"})
    await pfc.process(input_sig)
    assert len(pfc.goal_stack) > 0

    # Strategy switch should clear
    switch_sig = Signal(type=SignalType.STRATEGY_SWITCH, source="acc",
                        payload={"reason": "errors"})
    result = await pfc.process(switch_sig)
    assert result is None
    assert pfc.goal_stack == []


async def test_returns_none_for_unhandled_signal(pfc):
    sig = Signal(type=SignalType.ENCODE, source="memory",
                 payload={"content": "data"})
    result = await pfc.process(sig)
    assert result is None


async def test_name_and_position(pfc):
    assert pfc.name == "prefrontal_cortex"
    assert pfc.position.x == 0
    assert pfc.position.y == 60
    assert pfc.position.z == 20


async def test_calls_llm_when_provider_present(pfc_with_llm, mock_llm_provider):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Hello, how are you?"})
    result = await pfc_with_llm.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    mock_llm_provider.chat.assert_awaited_once()
    action = result.payload["actions"][0]
    assert action["args"]["text"] == "Hello! I'm doing well."


async def test_stub_behavior_without_provider(pfc):
    """Existing stub behavior preserved when no provider."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    result = await pfc.process(sig)
    assert result.payload["actions"][0]["args"]["text"] == "Processing: test"


async def test_llm_error_falls_back_to_stub(pfc_with_llm, mock_llm_provider):
    mock_llm_provider.chat.return_value = LLMResponse(
        content=None, finish_reason="error", usage={"error": "API timeout"}
    )
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    result = await pfc_with_llm.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    # Falls back to stub response
    assert "Processing:" in result.payload["actions"][0]["args"]["text"]


# ── _parse_entities tests (Eichenbaum 2000) ──────────────────────────


def test_parse_entities_extracts_valid_block():
    """Valid <entities> block is parsed and stripped from response."""
    response = (
        'Here is my answer.\n\n'
        '<entities>\n'
        '{"entities": ["Python", "AI"], "relations": [["Python", "used_for", "AI"]]}\n'
        '</entities>'
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Here is my answer."
    assert data["entities"] == ["Python", "AI"]
    assert data["relations"] == [["Python", "used_for", "AI"]]


def test_parse_entities_no_block():
    """Response without <entities> block returns original text and empty entities."""
    response = "Just a normal response."
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Just a normal response."
    assert data["entities"] == []
    assert data["about_user"] == []
    assert data["knowledge"] == []


def test_parse_entities_malformed_json():
    """Malformed JSON inside <entities> block returns empty entities."""
    response = (
        "Some response.\n"
        "<entities>\n"
        "{not valid json}\n"
        "</entities>"
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Some response."
    assert data["entities"] == []
    assert data["about_user"] == []


def test_parse_entities_missing_keys():
    """JSON without expected keys gets defaults added."""
    response = (
        "Answer.\n"
        '<entities>\n'
        '{"entities": ["X"]}\n'
        '</entities>'
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Answer."
    assert data["entities"] == ["X"]
    assert data["relations"] == []


def test_parse_entities_strips_trailing_whitespace():
    """Whitespace between response text and entities block is removed."""
    response = (
        "My response.   \n\n\n"
        '<entities>\n'
        '{"entities": [], "relations": []}\n'
        '</entities>'
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "My response."


def test_parse_entities_multiline_json():
    """Multiline JSON inside entities block is parsed correctly."""
    response = (
        "Response text.\n"
        "<entities>\n"
        '{\n'
        '  "entities": ["A", "B"],\n'
        '  "relations": [["A", "connects", "B"]]\n'
        '}\n'
        "</entities>"
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Response text."
    assert data["entities"] == ["A", "B"]
    assert len(data["relations"]) == 1


def test_parse_entities_5_element():
    """_parse_entities should handle [s, r, t, confidence, category] format."""
    response = (
        'Hello!\n<entities>\n'
        '{"entities": ["coffee", "user"], '
        '"relations": [["user", "like", "coffee", 0.9, "PREFERENCE"]]}\n'
        '</entities>'
    )
    clean, data = PrefrontalCortex._parse_entities(response)
    assert clean == "Hello!"
    assert data["relations"] == [["user", "like", "coffee", 0.9, "PREFERENCE"]]


def test_parse_entities_mixed_lengths():
    """Parser should accept 3, 4, or 5-element relations."""
    response = (
        'Hi\n<entities>\n'
        '{"entities": [], "relations": ['
        '["a", "r", "b"],'
        '["c", "r", "d", 0.8],'
        '["e", "r", "f", 0.9, "ACTION"]'
        ']}\n</entities>'
    )
    _, data = PrefrontalCortex._parse_entities(response)
    assert len(data["relations"]) == 3


async def test_llm_response_with_entities_stripped_from_output(pfc_with_llm, mock_llm_provider):
    """When LLM returns entities block, user-facing text should not contain it."""
    mock_llm_provider.chat.return_value = LLMResponse(
        content=(
            "Python is great for AI.\n\n"
            "<entities>\n"
            '{"entities": ["Python", "AI"], "relations": [["Python", "used_for", "AI"]]}\n'
            "</entities>"
        ),
        finish_reason="stop",
        usage={},
    )
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Tell me about Python"})
    result = await pfc_with_llm.process(sig)
    action_text = result.payload["actions"][0]["args"]["text"]
    assert "<entities>" not in action_text
    assert action_text == "Python is great for AI."
    # Entities should be in signal metadata
    extracted = result.metadata.get("extracted_entities", {})
    assert extracted["entities"] == ["Python", "AI"]
    assert extracted["relations"] == [["Python", "used_for", "AI"]]


async def test_no_llm_provider_returns_empty_entities(pfc):
    """Without LLM provider, extracted_entities should be empty defaults."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "hello"})
    result = await pfc.process(sig)
    extracted = result.metadata.get("extracted_entities", {})
    assert extracted["entities"] == []
    assert extracted["relations"] == []


# ── GoalTree tests (Koechlin 2003) ─────────────────────────────────

from brain_agent.regions.goal_tree import GoalTree, GoalNode


class TestGoalNode:
    def test_add_subgoal_creates_child(self):
        root = GoalNode(description="Build app", level="rostral")
        child = root.add_subgoal("Design UI", level="mid")
        assert child.parent_id == root.id
        assert child in root.children
        assert child.level == "mid"
        assert child.status == "active"

    def test_get_active_leaf_returns_deepest_active(self):
        root = GoalNode(description="Build app", level="rostral")
        mid = root.add_subgoal("Design UI", level="mid")
        leaf = mid.add_subgoal("Sketch wireframe", level="caudal")
        assert root.get_active_leaf() is leaf

    def test_get_active_leaf_skips_completed_children(self):
        root = GoalNode(description="Build app", level="rostral")
        child1 = root.add_subgoal("Step 1", level="caudal")
        child1.status = "completed"
        child2 = root.add_subgoal("Step 2", level="caudal")
        assert root.get_active_leaf() is child2

    def test_get_active_leaf_returns_self_when_no_active_children(self):
        root = GoalNode(description="Solo goal", level="rostral")
        assert root.get_active_leaf() is root

    def test_get_active_leaf_returns_none_when_completed(self):
        node = GoalNode(description="Done", level="caudal", status="completed")
        assert node.get_active_leaf() is None

    def test_complete_cascades_to_children(self):
        root = GoalNode(description="Root", level="rostral")
        c1 = root.add_subgoal("Child 1")
        c2 = root.add_subgoal("Child 2")
        root.complete()
        assert root.status == "completed"
        assert c1.status == "completed"
        assert c2.status == "completed"

    def test_complete_does_not_affect_already_failed(self):
        root = GoalNode(description="Root", level="rostral")
        c1 = root.add_subgoal("Already failed")
        c1.status = "failed"
        root.complete()
        assert root.status == "completed"
        assert c1.status == "failed"  # Not overwritten

    def test_to_context_produces_readable_output(self):
        root = GoalNode(description="Build app", level="rostral")
        mid = root.add_subgoal("Design UI", level="mid")
        mid.add_subgoal("Sketch wireframe", level="caudal")
        ctx = root.to_context()
        assert "[ABSTRACT] Build app" in ctx
        assert "  [ ] [SUB-GOAL] Design UI" in ctx
        assert "    [ ] [ACTION] Sketch wireframe" in ctx

    def test_to_context_marks_completed(self):
        node = GoalNode(description="Done task", level="caudal")
        node.complete()
        ctx = node.to_context()
        assert "[x]" in ctx


class TestGoalTree:
    def test_set_goal_adds_root(self):
        tree = GoalTree()
        node = tree.set_goal("Main goal")
        assert node in tree.roots
        assert node.level == "rostral"
        assert node.status == "active"

    def test_get_current_focus_returns_deepest_active(self):
        tree = GoalTree()
        root = tree.set_goal("Main goal")
        sub = root.add_subgoal("Sub goal", level="mid")
        leaf = sub.add_subgoal("Do thing", level="caudal")
        assert tree.get_current_focus() is leaf

    def test_get_current_focus_returns_none_when_empty(self):
        tree = GoalTree()
        assert tree.get_current_focus() is None

    def test_get_current_focus_skips_completed_roots(self):
        tree = GoalTree()
        old = tree.set_goal("Old goal")
        old.complete()
        new = tree.set_goal("New goal")
        assert tree.get_current_focus() is new

    def test_to_context_shows_active_roots(self):
        tree = GoalTree()
        tree.set_goal("Goal A")
        tree.set_goal("Goal B")
        ctx = tree.to_context()
        assert "Goal A" in ctx
        assert "Goal B" in ctx

    def test_to_context_empty_tree(self):
        tree = GoalTree()
        assert tree.to_context() == "No active goals."

    def test_to_context_hides_completed_roots(self):
        tree = GoalTree()
        old = tree.set_goal("Old")
        old.complete()
        tree.set_goal("Active")
        ctx = tree.to_context()
        assert "Old" not in ctx
        assert "Active" in ctx

    def test_clear_removes_all(self):
        tree = GoalTree()
        tree.set_goal("A")
        tree.set_goal("B")
        tree.clear()
        assert tree.roots == []
        assert tree.to_context() == "No active goals."


class TestPFCGoalTreeIntegration:
    async def test_pfc_uses_goal_tree(self):
        """PFC should use GoalTree internally."""
        pfc = PrefrontalCortex()
        assert isinstance(pfc.goals, GoalTree)

    async def test_goal_stack_backward_compatible(self):
        """goal_stack property should return flat list of active goals."""
        pfc = PrefrontalCortex()
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "analyze code"})
        await pfc.process(sig)
        assert pfc.goal_stack == ["analyze code"]
        assert pfc.goals.get_current_focus().description == "analyze code"

    async def test_strategy_switch_clears_goal_tree(self):
        """Strategy switch should clear the entire goal tree."""
        pfc = PrefrontalCortex()
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "task"})
        await pfc.process(sig)
        assert len(pfc.goals.roots) > 0

        switch = Signal(type=SignalType.STRATEGY_SWITCH, source="acc",
                        payload={"reason": "errors"})
        await pfc.process(switch)
        assert pfc.goals.roots == []
        assert pfc.goal_stack == []

    async def test_goal_tree_context_in_llm_call(self, mock_llm_provider):
        """LLM system prompt should include goal tree context."""
        pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
        sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                     payload={"text": "build a web app"})
        await pfc.process(sig)
        # Check that the system message includes goal context
        call_args = mock_llm_provider.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Active Goals" in system_msg or "Current goals:" in system_msg
        assert "build a web app" in system_msg


# ── Metacognition tests (Fleming & Dolan 2012) ──────────────────────


def test_parse_confidence_from_response():
    pfc = PrefrontalCortex()
    text = 'Some answer here\n<meta>{"confidence": 0.85}</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.85
    assert "<meta>" not in clean
    assert "Some answer here" in clean


def test_parse_confidence_missing():
    pfc = PrefrontalCortex()
    text = "Just a plain answer"
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.7
    assert clean == "Just a plain answer"


def test_parse_confidence_low():
    pfc = PrefrontalCortex()
    text = 'Not sure\n<meta>{"confidence": 0.3}</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.3


def test_parse_confidence_clamp():
    pfc = PrefrontalCortex()
    text = 'Over\n<meta>{"confidence": 1.5}</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 1.0


def test_parse_confidence_invalid_json():
    pfc = PrefrontalCortex()
    text = 'Answer\n<meta>not json</meta>'
    confidence, clean = pfc._parse_metacognition(text)
    assert confidence == 0.7  # Default on parse error


async def test_metacognition_in_signal_metadata(mock_llm_provider):
    """Metacognition confidence should be stored in plan signal metadata."""
    mock_llm_provider.chat.return_value = LLMResponse(
        content='Hello!\n<meta>{"confidence": 0.9}</meta>',
        finish_reason="stop",
        usage={},
    )
    pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Hi there"})
    result = await pfc.process(sig)
    assert result.metadata["metacognition"]["confidence"] == 0.9
    # Meta tag should be stripped from user-facing text
    action_text = result.payload["actions"][0]["args"]["text"]
    assert "<meta>" not in action_text


async def test_metacognition_instruction_in_system_prompt(mock_llm_provider):
    """System prompt should include metacognition instruction."""
    pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    await pfc.process(sig)
    call_args = mock_llm_provider.chat.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "metacognitive self-assessment" in system_msg
    assert "<meta>" in system_msg


# ── GABA neuromodulator tests (Buzsaki 2006) ─────────────────────────


def test_describe_neuromodulator_state_includes_gaba():
    pfc = PrefrontalCortex()
    nm = {"dopamine": 0.5, "norepinephrine": 0.5, "serotonin": 0.5,
          "acetylcholine": 0.5, "cortisol": 0.5, "epinephrine": 0.5, "gaba": 0.8}
    desc = pfc._describe_neuromodulator_state(nm)
    assert "GABA" in desc or "inhibitory" in desc.lower()


def test_describe_neuromodulator_state_low_gaba():
    pfc = PrefrontalCortex()
    nm = {"dopamine": 0.5, "norepinephrine": 0.5, "serotonin": 0.5,
          "acetylcholine": 0.5, "cortisol": 0.5, "epinephrine": 0.5, "gaba": 0.2}
    desc = pfc._describe_neuromodulator_state(nm)
    assert "GABA" in desc or "disinhibited" in desc.lower() or "inhibitory" in desc.lower()


# ── Interoceptive state tests (Craig 2009) ────────────────────────────


async def test_interoceptive_state_in_system_prompt(mock_llm_provider):
    """Insula interoceptive state should appear in PFC system prompt."""
    pfc = PrefrontalCortex(llm_provider=mock_llm_provider)
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    sig.metadata["upstream_context"] = {
        "interoceptive_state": {
            "stress_level": 0.7,
            "energy_level": 0.4,
            "emotional_awareness": 0.6,
            "risk_sensitivity": 0.8,
        },
    }
    await pfc.process(sig)
    call_args = mock_llm_provider.chat.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "Interoceptive State" in system_msg
    assert "Stress level" in system_msg
    assert "Energy level" in system_msg
    assert "Risk sensitivity" in system_msg
