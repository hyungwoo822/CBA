from brain_agent.memory.semantic_extractor import (
    build_extraction_prompt,
    parse_extraction_response,
)


def test_parse_5_element_relations():
    """Parser should accept 5-element relations from new prompt format."""
    response = (
        '<fact>User enjoys coffee regularly</fact>\n'
        '<relations>[["user", "drink", "coffee", 0.9, "PREFERENCE"],'
        '["coffee", "contain", "caffeine", 0.8, "ATTRIBUTE"]]</relations>'
    )
    fact, relations = parse_extraction_response(response)
    assert fact == "User enjoys coffee regularly"
    assert len(relations) == 2
    assert relations[0] == ["user", "drink", "coffee", 0.9, "PREFERENCE"]


def test_parse_mixed_element_relations():
    """Parser should accept 3, 4, or 5-element relations."""
    response = (
        '<fact>fact</fact>\n'
        '<relations>[["a","r","b"],["c","r","d",0.8],["e","r","f",0.9,"ACTION"]]</relations>'
    )
    _, relations = parse_extraction_response(response)
    assert len(relations) == 3


def test_extraction_prompt_contains_normalization_rules():
    """Prompt should instruct English normalization."""
    prompt = build_extraction_prompt([{"content": "test memory"}])
    assert "English lowercase" in prompt
    assert "verb infinitive" in prompt
