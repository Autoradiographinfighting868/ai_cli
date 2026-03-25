"""Tests for jicli/prompts/builder.py — Template-based prompt building."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jicli.prompts.builder import build_system_prompt, build_reprime_prompt


def test_build_system_prompt():
    """System prompt should include tool definitions and context."""
    tool_defs = [
        {"name": "Bash", "description": "Run shell commands", "parameters": {"properties": {"command": {"description": "The command"}}, "required": ["command"]}},
        {"name": "Read", "description": "Read files", "parameters": {"properties": {"file_path": {"description": "Path"}}, "required": ["file_path"]}},
    ]
    prompt = build_system_prompt(
        tool_definitions=tool_defs,
        cwd="/tmp/test",
        model="qwen3.5:4b",
    )
    assert "Bash" in prompt
    assert "Read" in prompt
    assert "/tmp/test" in prompt
    assert len(prompt) > 100
    print("  PASS: build_system_prompt")


def test_build_system_prompt_with_memory():
    """System prompt should include memory block when provided."""
    tool_defs = [{"name": "Bash", "description": "Run commands", "parameters": {}}]
    prompt = build_system_prompt(
        tool_definitions=tool_defs,
        cwd="/tmp",
        model="test-model",
        memory_block="User prefers dark mode",
    )
    assert "dark mode" in prompt
    print("  PASS: build_system_prompt with memory")


def test_build_system_prompt_with_errors():
    """System prompt should include error context when provided."""
    tool_defs = [{"name": "Bash", "description": "Run commands", "parameters": {}}]
    prompt = build_system_prompt(
        tool_definitions=tool_defs,
        cwd="/tmp",
        model="test-model",
        error_context="Common issue: file not found when path is relative",
    )
    assert "file not found" in prompt
    print("  PASS: build_system_prompt with errors")


def test_build_reprime_prompt():
    """Re-prime prompt should contain session context."""
    prompt = build_reprime_prompt(
        session_summary="Working on a CLI tool",
        key_facts=["Python", "Ollama API"],
        accomplishments="Built config module, tool system",
        current_task="Write tests and README",
    )
    assert "CLI tool" in prompt
    assert "Python" in prompt
    assert "tests" in prompt.lower() or "README" in prompt
    print("  PASS: build_reprime_prompt")


def test_templates_exist():
    """Template files should exist."""
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TEMPLATES")
    assert os.path.exists(os.path.join(template_dir, "system.md")), "system.md missing"
    assert os.path.exists(os.path.join(template_dir, "reprime.md")), "reprime.md missing"
    assert os.path.exists(os.path.join(template_dir, "planner.md")), "planner.md missing"
    print("  PASS: templates exist")


if __name__ == "__main__":
    print("=== Prompt Tests ===")
    test_build_system_prompt()
    test_build_system_prompt_with_memory()
    test_build_system_prompt_with_errors()
    test_build_reprime_prompt()
    test_templates_exist()
    print("\nAll prompt tests passed!")
