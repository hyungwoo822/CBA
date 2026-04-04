def test_cli_imports():
    from brain_agent.cli.commands import main, print_help
    assert callable(main)


def test_version_accessible():
    from brain_agent import __version__, BrainAgent
    assert __version__ == "0.1.0"
    assert BrainAgent is not None
